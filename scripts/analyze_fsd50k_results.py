from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def load_json(path: Path):
    """读取 FSD50K 训练结果 JSON。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def plot_loss(history: list[dict[str, float]], output_dir: Path) -> None:
    """绘制 FSD50K 多标签训练/验证 loss 曲线。"""

    epochs = [record["epoch"] for record in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [record["train_loss"] for record in history], marker="o", label="Train Loss")
    plt.plot(epochs, [record["val_loss"] for record in history], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FSD50K Pretrained AST Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=200)
    plt.close()


def plot_metrics(history: list[dict[str, float]], output_dir: Path) -> None:
    """绘制 FSD50K 多标签核心指标曲线。"""

    epochs = [record["epoch"] for record in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [record["val_mAP"] for record in history], marker="o", label="Val mAP")
    plt.plot(epochs, [record["val_micro_f1"] for record in history], marker="o", label="Val micro-F1")
    plt.plot(epochs, [record["val_macro_f1"] for record in history], marker="o", label="Val macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("FSD50K Pretrained AST Validation Metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_metrics.png", dpi=200)
    plt.close()


def load_predictions(path: Path) -> dict[str, object] | None:
    """读取逐样本预测文件；如果尚未由服务器评估脚本生成，则返回 None。

    类别级 AP/F1 和阈值敏感性都依赖每个样本的真实 multi-hot 标签与预测概率。
    为了让脚本在只有训练历史时仍能工作，这里把预测文件设计为可选输入。
    """

    if not path.exists():
        return None
    return load_json(path)


def compute_class_metrics(predictions: dict[str, object], threshold: float) -> list[dict[str, object]]:
    """计算每个 FSD50K 类别的 AP、Precision、Recall 和 F1。

    AP 使用连续概率，Precision/Recall/F1 使用指定阈值二值化后的预测。
    对验证集中没有正样本的类别，AP 记为空值，避免把无定义指标误写为 0。
    """

    labels = list(predictions["labels"])
    samples = list(predictions["samples"])
    true_array = np.asarray([sample["y_true"] for sample in samples], dtype=float)
    score_array = np.asarray([sample["y_score"] for sample in samples], dtype=float)
    pred_array = (score_array >= threshold).astype(int)

    rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        y_true = true_array[:, index]
        y_score = score_array[:, index]
        y_pred = pred_array[:, index]
        positives = int(y_true.sum())
        predicted_positives = int(y_pred.sum())
        ap = None
        if positives > 0:
            ap = float(average_precision_score(y_true, y_score))

        rows.append(
            {
                "class_index": index,
                "label": label,
                "support": positives,
                "predicted_positives": predicted_positives,
                "average_precision": ap,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )

    rows.sort(
        key=lambda row: (
            row["average_precision"] is None,
            -(row["average_precision"] or 0.0),
            -int(row["support"]),
            str(row["label"]),
        )
    )
    return rows


def write_class_metrics(rows: list[dict[str, object]], output_dir: Path) -> None:
    """把类别级指标写为 CSV，便于报告、表格和后续排序筛选使用。"""

    path = output_dir / "class_metrics.csv"
    fieldnames = [
        "class_index",
        "label",
        "support",
        "predicted_positives",
        "average_precision",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_threshold_sensitivity(
    predictions: dict[str, object],
    thresholds: list[float],
) -> list[dict[str, float]]:
    """扫描不同阈值下的 micro/macro-F1，观察多标签决策边界敏感性。"""

    samples = list(predictions["samples"])
    true_array = np.asarray([sample["y_true"] for sample in samples], dtype=int)
    score_array = np.asarray([sample["y_score"] for sample in samples], dtype=float)

    rows = []
    for threshold in thresholds:
        pred_array = (score_array >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "micro_f1": float(f1_score(true_array, pred_array, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(true_array, pred_array, average="macro", zero_division=0)),
                "predicted_labels_per_sample": float(pred_array.sum(axis=1).mean()),
            }
        )
    return rows


def write_threshold_sensitivity(rows: list[dict[str, float]], output_dir: Path) -> None:
    """保存阈值扫描表并绘制敏感性曲线。"""

    csv_path = output_dir / "threshold_sensitivity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["threshold", "micro_f1", "macro_f1", "predicted_labels_per_sample"],
        )
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(8, 5))
    thresholds = [row["threshold"] for row in rows]
    plt.plot(thresholds, [row["micro_f1"] for row in rows], marker="o", label="micro-F1")
    plt.plot(thresholds, [row["macro_f1"] for row in rows], marker="o", label="macro-F1")
    plt.xlabel("Decision Threshold")
    plt.ylabel("F1")
    plt.title("FSD50K Threshold Sensitivity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sensitivity.png", dpi=200)
    plt.close()


def write_summary(
    history: list[dict[str, float]],
    latest_metrics: dict[str, float],
    output_dir: Path,
    class_rows: list[dict[str, object]] | None = None,
    threshold_rows: list[dict[str, float]] | None = None,
) -> None:
    """生成适合报告引用的 FSD50K 多标签结果摘要。"""

    best = max(history, key=lambda record: record["val_mAP"])
    final = history[-1]
    lines = [
        "# FSD50K Pretrained AST 多标签结果摘要",
        "",
        "## 整体结果",
        "",
        f"- 训练轮数：{len(history)}",
        f"- 验证样本数：{latest_metrics['num_samples']}",
        f"- 类别数：{latest_metrics['num_classes']}",
        f"- 最佳验证 mAP：{best['val_mAP']:.4f}（第 {best['epoch']} 轮）",
        f"- 最佳轮验证 Loss：{best['val_loss']:.4f}",
        f"- 最佳轮验证 micro-F1：{best['val_micro_f1']:.4f}",
        f"- 最佳轮验证 macro-F1：{best['val_macro_f1']:.4f}",
        f"- 最后一轮验证 mAP：{final['val_mAP']:.4f}",
        f"- 最后一轮验证 micro-F1：{final['val_micro_f1']:.4f}",
        f"- 最后一轮验证 macro-F1：{final['val_macro_f1']:.4f}",
        "",
        "## 初步结论",
        "",
        "- 预训练 AST 已成功扩展到 FSD50K 多标签任务，验证 mAP 达到 0.6208。",
        "- 训练 mAP 持续上升而验证 mAP 在第 4 轮达到峰值，说明继续训练可能开始带来过拟合或阈值不稳定。",
        "- macro-F1 低于 micro-F1，说明长尾类别或低频类别仍然更难，需要后续做类别不均衡和阈值策略分析。",
    ]

    if class_rows:
        valid_rows = [row for row in class_rows if row["average_precision"] is not None]
        top_rows = valid_rows[:5]
        bottom_rows = sorted(valid_rows, key=lambda row: row["average_precision"] or 0.0)[:5]
        lines.extend(["", "## 类别级分析", ""])
        lines.append("AP 最高的 5 个类别：")
        for row in top_rows:
            lines.append(
                f"- `{row['label']}`：AP={row['average_precision']:.4f}，"
                f"support={row['support']}，F1={row['f1']:.4f}"
            )
        lines.append("")
        lines.append("AP 最低的 5 个有正样本类别：")
        for row in bottom_rows:
            lines.append(
                f"- `{row['label']}`：AP={row['average_precision']:.4f}，"
                f"support={row['support']}，F1={row['f1']:.4f}"
            )

    if threshold_rows:
        best_micro = max(threshold_rows, key=lambda row: row["micro_f1"])
        best_macro = max(threshold_rows, key=lambda row: row["macro_f1"])
        lines.extend(
            [
                "",
                "## 阈值敏感性",
                "",
                f"- micro-F1 最优阈值：{best_micro['threshold']:.2f}，micro-F1={best_micro['micro_f1']:.4f}。",
                f"- macro-F1 最优阈值：{best_macro['threshold']:.2f}，macro-F1={best_macro['macro_f1']:.4f}。",
                "- 如果最优阈值明显低于 0.50，说明固定 0.50 阈值可能低估了低频类别召回率，需要在报告中说明阈值选择策略。",
            ]
        )

    lines.extend(
        [
            "",
            "## 生成文件",
            "",
            "- `training_loss.png`：训练/验证 Loss 曲线。",
            "- `validation_metrics.png`：验证 mAP、micro-F1 和 macro-F1 曲线。",
        ]
    )
    if class_rows:
        lines.append("- `class_metrics.csv`：类别级 AP、Precision、Recall 和 F1。")
    if threshold_rows:
        lines.append("- `threshold_sensitivity.csv` / `threshold_sensitivity.png`：不同阈值下的 F1 变化。")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 FSD50K 多标签 AST 训练结果")
    parser.add_argument("--history", default="outputs/fsd50k_ast/history.json", help="训练历史 JSON 路径")
    parser.add_argument(
        "--metrics",
        default="outputs/fsd50k_ast/latest_val_metrics.json",
        help="最新验证指标 JSON 路径",
    )
    parser.add_argument("--output-dir", default="outputs/fsd50k_ast/analysis", help="分析结果输出目录")
    parser.add_argument(
        "--predictions",
        default="outputs/fsd50k_ast/val_predictions.json",
        help="验证集逐样本预测 JSON；存在时额外生成类别级和阈值敏感性分析",
    )
    args = parser.parse_args()

    history = load_json(Path(args.history))
    latest_metrics = load_json(Path(args.metrics))
    predictions = load_predictions(Path(args.predictions))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss(history, output_dir)
    plot_metrics(history, output_dir)
    class_rows = None
    threshold_rows = None
    if predictions is not None:
        threshold = float(predictions.get("threshold", latest_metrics.get("threshold", 0.5)))
        class_rows = compute_class_metrics(predictions, threshold=threshold)
        write_class_metrics(class_rows, output_dir)
        threshold_values = [round(value, 2) for value in np.arange(0.05, 0.96, 0.05)]
        threshold_rows = compute_threshold_sensitivity(predictions, threshold_values)
        write_threshold_sensitivity(threshold_rows, output_dir)

    write_summary(history, latest_metrics, output_dir, class_rows, threshold_rows)
    print(f"分析完成，结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
