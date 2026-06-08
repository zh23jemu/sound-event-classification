from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sound_event_classification.config import load_config
from sound_event_classification.data import ESC50Dataset
from sound_event_classification.metrics import classification_metrics


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子，降低小数据集实验波动。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """根据配置选择训练设备；`auto` 会优先使用 CUDA。"""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_label_maps(dataset: ESC50Dataset, num_classes: int) -> tuple[dict[int, str], dict[str, int]]:
    """从 ESC-50 元数据中构建 id2label / label2id，保存进 Hugging Face 模型配置。

    如果某些类别在当前 fold 中缺失，则用 `class_xx` 占位。ESC-50 的训练 fold
    通常覆盖全部 50 类，这里的兜底主要是为了让脚本面对其它划分时也能稳定运行。
    """

    id_to_category = {item.target: item.category for item in dataset.items}
    id2label = {label_id: id_to_category.get(label_id, f"class_{label_id:02d}") for label_id in range(num_classes)}
    label2id = {label: label_id for label_id, label in id2label.items()}
    return id2label, label2id


@dataclass
class AstBatchCollator:
    """把 ESC-50 waveform batch 转成 AST 模型需要的 `input_values`。

    Hugging Face AST feature extractor 接收一组单声道 waveform，并在内部完成频谱、
    归一化、padding/truncation 等处理。这里把张量转回 NumPy list，是为了兼容
    feature extractor 的标准接口；真正的模型前向仍在 GPU 上执行。
    """

    feature_extractor: object
    sample_rate: int

    def __call__(self, batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms = [waveform.squeeze(0).numpy() for waveform, _ in batch]
        targets = torch.tensor([target for _, target in batch], dtype=torch.long)
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        return inputs["input_values"], targets


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, list[int], list[int]]:
    """执行一个 AST 训练或验证 epoch。

    训练阶段使用模型自带的交叉熵 loss；验证阶段不更新参数，只收集预测结果。
    `gradient_accumulation_steps` 用于在显存紧张时模拟更大的 batch。
    """

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, (input_values, target) in enumerate(tqdm(dataloader, leave=False), start=1):
        input_values = input_values.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
                outputs = model(input_values=input_values, labels=target)
                raw_loss = outputs.loss
                loss = raw_loss / gradient_accumulation_steps

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                should_step = step % gradient_accumulation_steps == 0 or step == len(dataloader)
                if should_step:
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = target.size(0)
        total_loss += float(raw_loss.detach().cpu()) * batch_size
        total_samples += batch_size
        y_true.extend(target.detach().cpu().tolist())
        y_pred.extend(outputs.logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / max(total_samples, 1), y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="微调预训练 AST 模型进行 ESC-50 分类")
    parser.add_argument("--config", default="configs/esc50_ast.yaml", help="AST 训练配置文件")
    args = parser.parse_args()

    # Transformers 只在 AST 脚本实际运行时才需要，避免 CNN baseline 的本地检查被额外依赖阻塞。
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except ImportError as exc:
        raise ImportError(
            "缺少 transformers 依赖。请在项目 .venv 中运行："
            ".venv/bin/python -m pip install transformers accelerate"
        ) from exc

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    device = resolve_device(config["train"]["device"])
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ESC50Dataset(
        root=config["data"]["root"],
        folds=list(config["data"]["train_folds"]),
        sample_rate=int(config["data"]["sample_rate"]),
        duration_seconds=float(config["data"]["duration_seconds"]),
    )
    val_dataset = ESC50Dataset(
        root=config["data"]["root"],
        folds=list(config["data"]["val_folds"]),
        sample_rate=int(config["data"]["sample_rate"]),
        duration_seconds=float(config["data"]["duration_seconds"]),
    )

    pretrained_name = str(config["model"]["pretrained_name"])
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_name)
    collator = AstBatchCollator(
        feature_extractor=feature_extractor,
        sample_rate=int(config["data"]["sample_rate"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=collator,
    )

    id2label, label2id = build_label_maps(train_dataset, int(config["model"]["num_classes"]))
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_name,
        num_labels=int(config["model"]["num_classes"]),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    amp_enabled = bool(config["train"].get("amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    gradient_accumulation_steps = max(int(config["train"].get("gradient_accumulation_steps", 1)), 1)

    history = []
    best_accuracy = -1.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_loss, train_true, train_pred = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        val_loss, val_true, val_pred = run_epoch(
            model,
            val_loader,
            device,
            amp_enabled=amp_enabled,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        train_metrics = classification_metrics(train_true, train_pred)
        val_metrics = classification_metrics(val_true, val_pred)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if float(val_metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(val_metrics["accuracy"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

        with (output_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)
        with (output_dir / "latest_val_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(val_metrics, file, ensure_ascii=False, indent=2)

    print(f"训练完成，最佳验证 Accuracy：{best_accuracy:.4f}")
    print(f"最佳模型保存到：{best_path}")


if __name__ == "__main__":
    main()
