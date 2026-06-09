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
from sound_event_classification.data import FSD50KDataset
from sound_event_classification.metrics import multilabel_classification_metrics


def set_seed(seed: int) -> None:
    """固定随机种子，减少 FSD50K 微调结果波动。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """根据配置选择训练设备。"""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class Fsd50kAstBatchCollator:
    """把 FSD50K waveform batch 转换为 AST 输入。"""

    feature_extractor: object
    sample_rate: int

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms = [waveform.squeeze(0).numpy() for waveform, _ in batch]
        targets = torch.stack([target for _, target in batch]).float()
        inputs = self.feature_extractor(waveforms, sampling_rate=self.sample_rate, return_tensors="pt")
        return inputs["input_values"], targets


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, list[list[float]], list[list[float]]]:
    """执行一个 FSD50K 多标签训练或验证 epoch。"""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    y_true: list[list[float]] = []
    y_score: list[list[float]] = []

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
        y_score.extend(torch.sigmoid(outputs.logits).detach().cpu().tolist())

    return total_loss / max(total_samples, 1), y_true, y_score


def main() -> None:
    parser = argparse.ArgumentParser(description="微调预训练 AST 模型进行 FSD50K 多标签分类")
    parser.add_argument("--config", default="configs/fsd50k_ast.yaml", help="FSD50K AST 训练配置文件")
    args = parser.parse_args()

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

    train_dataset = FSD50KDataset(
        root=config["data"]["root"],
        split=str(config["data"]["train_split"]),
        sample_rate=int(config["data"]["sample_rate"]),
        duration_seconds=float(config["data"]["duration_seconds"]),
    )
    val_dataset = FSD50KDataset(
        root=config["data"]["root"],
        split=str(config["data"]["val_split"]),
        sample_rate=int(config["data"]["sample_rate"]),
        duration_seconds=float(config["data"]["duration_seconds"]),
    )

    pretrained_name = str(config["model"]["pretrained_name"])
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_name)
    collator = Fsd50kAstBatchCollator(feature_extractor, int(config["data"]["sample_rate"]))

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

    id2label = {index: label for index, label in enumerate(train_dataset.index_to_label)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_name,
        num_labels=int(config["model"]["num_classes"]),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
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
    threshold = float(config["train"].get("threshold", 0.5))

    history = []
    best_map = -1.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_loss, train_true, train_score = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        val_loss, val_true, val_score = run_epoch(
            model,
            val_loader,
            device,
            amp_enabled=amp_enabled,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        train_metrics = multilabel_classification_metrics(train_true, train_score, threshold=threshold)
        val_metrics = multilabel_classification_metrics(val_true, val_score, threshold=threshold)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mAP": train_metrics["mAP"],
            "val_mAP": val_metrics["mAP"],
            "train_micro_f1": train_metrics["micro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if float(val_metrics["mAP"]) > best_map:
            best_map = float(val_metrics["mAP"])
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

    print(f"训练完成，最佳验证 mAP：{best_map:.4f}")
    print(f"最佳模型保存到：{best_path}")


if __name__ == "__main__":
    main()
