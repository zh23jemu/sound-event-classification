from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from train_fsd50k_ast import Fsd50kAstBatchCollator, resolve_device


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[list[list[float]], list[list[float]]]:
    """在验证集上保存真值和 sigmoid 概率。

    该脚本用于训练完成后的补充分析，因此不计算梯度，也不更新模型参数。
    返回的 `y_true` 和 `y_score` 会被写入 JSON，后续分析脚本据此计算每类
    AP/F1、不同阈值下的 micro/macro-F1 等指标。
    """

    model.eval()
    y_true: list[list[float]] = []
    y_score: list[list[float]] = []

    with torch.no_grad():
        for input_values, target in tqdm(dataloader, leave=False):
            input_values = input_values.to(device)
            target = target.to(device)

            with torch.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
                outputs = model(input_values=input_values)

            y_true.extend(target.detach().cpu().tolist())
            y_score.extend(torch.sigmoid(outputs.logits).detach().cpu().tolist())

    return y_true, y_score


def write_predictions(
    path: Path,
    dataset: FSD50KDataset,
    y_true: list[list[float]],
    y_score: list[list[float]],
    metrics: dict[str, object],
) -> None:
    """写出验证集逐样本预测，保持类别顺序和样本顺序可追溯。"""

    samples = []
    for item, true_row, score_row in zip(dataset.samples, y_true, y_score):
        samples.append(
            {
                "fname": item.fname,
                "labels": item.labels,
                "y_true": true_row,
                "y_score": score_row,
            }
        )

    payload = {
        "split": dataset.split,
        "threshold": metrics["threshold"],
        "labels": dataset.index_to_label,
        "metrics": metrics,
        "samples": samples,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 FSD50K AST 最佳模型并导出逐样本预测概率")
    parser.add_argument("--config", default="configs/fsd50k_ast.yaml", help="FSD50K AST 配置文件")
    parser.add_argument("--checkpoint", default="outputs/fsd50k_ast/best_model.pt", help="最佳模型 checkpoint")
    parser.add_argument("--output", default="outputs/fsd50k_ast/val_predictions.json", help="预测概率输出 JSON")
    args = parser.parse_args()

    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except ImportError as exc:
        raise ImportError(
            "缺少 transformers 依赖。请在项目 .venv 中运行："
            ".venv/bin/python -m pip install transformers accelerate"
        ) from exc

    config = load_config(args.config)
    device = resolve_device(config["train"]["device"])
    threshold = float(config["train"].get("threshold", 0.5))
    amp_enabled = bool(config["train"].get("amp", False))

    val_dataset = FSD50KDataset(
        root=config["data"]["root"],
        split=str(config["data"]["val_split"]),
        sample_rate=int(config["data"]["sample_rate"]),
        duration_seconds=float(config["data"]["duration_seconds"]),
    )

    pretrained_name = str(config["model"]["pretrained_name"])
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_name)
    collator = Fsd50kAstBatchCollator(feature_extractor, int(config["data"]["sample_rate"]))
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=collator,
    )

    id2label = {index: label for index, label in enumerate(val_dataset.index_to_label)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_name,
        num_labels=int(config["model"]["num_classes"]),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_score = evaluate(model, val_loader, device, amp_enabled)
    metrics = multilabel_classification_metrics(y_true, y_score, threshold=threshold)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output_path, val_dataset, y_true, y_score, metrics)

    print(json.dumps(metrics, ensure_ascii=False))
    print(f"预测概率已保存到：{output_path}")


if __name__ == "__main__":
    main()
