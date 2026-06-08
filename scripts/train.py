from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sound_event_classification.config import load_config
from sound_event_classification.data import ESC50Dataset
from sound_event_classification.features import LogMelSpectrogram, SpecAugment
from sound_event_classification.metrics import classification_metrics
from sound_event_classification.models import build_model


def set_seed(seed: int) -> None:
    """固定随机种子，尽量保证实验结果可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """根据配置选择训练设备。"""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    augmentation: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int]]:
    """执行一个训练或验证 epoch。

    当 `optimizer` 为 None 时进入评估模式，不更新参数；否则执行反向传播。
    返回平均 loss、真实标签和预测标签，方便统一计算指标。
    """

    is_train = optimizer is not None
    model.train(is_train)
    feature_extractor.train(False)
    if augmentation is not None:
        augmentation.train(is_train)

    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for waveform, target in tqdm(dataloader, leave=False):
        waveform = waveform.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(is_train):
            features = feature_extractor(waveform)
            if is_train and augmentation is not None:
                features = augmentation(features)
            logits = model(features)
            loss = criterion(logits, target)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = target.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        y_true.extend(target.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / max(total_samples, 1), y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 ESC-50 声音事件分类 baseline")
    parser.add_argument("--config", default="configs/esc50_baseline.yaml", help="训练配置文件")
    args = parser.parse_args()

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
    )

    feature_extractor = LogMelSpectrogram(
        sample_rate=int(config["data"]["sample_rate"]),
        n_mels=int(config["data"]["n_mels"]),
        n_fft=int(config["data"]["n_fft"]),
        hop_length=int(config["data"]["hop_length"]),
    ).to(device)
    augmentation_config = config.get("augmentation", {})
    augmentation = None
    if bool(augmentation_config.get("enabled", False)):
        # SpecAugment 只改变训练时的频谱输入，不改变标签和验证流程。
        # 这里通过配置控制强度，便于后续做“无增强 vs 有增强”的公平对比。
        augmentation = SpecAugment(
            time_mask_param=int(augmentation_config.get("time_mask_param", 32)),
            freq_mask_param=int(augmentation_config.get("freq_mask_param", 16)),
            num_time_masks=int(augmentation_config.get("num_time_masks", 2)),
            num_freq_masks=int(augmentation_config.get("num_freq_masks", 2)),
            p=float(augmentation_config.get("p", 1.0)),
        ).to(device)
    model = build_model(
        name=str(config["model"]["name"]),
        num_classes=int(config["model"]["num_classes"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    history = []
    best_accuracy = -1.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_loss, train_true, train_pred = run_epoch(
            model, feature_extractor, train_loader, criterion, device, augmentation, optimizer
        )
        val_loss, val_true, val_pred = run_epoch(
            model, feature_extractor, val_loader, criterion, device, augmentation
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
