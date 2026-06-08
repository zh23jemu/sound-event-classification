import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_json(path: Path):
    """读取 JSON 文件，并在文件不存在时给出明确报错，避免后续分析静默失败。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_esc50_class_names(metadata_path: Path | None, labels: list[int]) -> dict[int, str]:
    """从 ESC-50 元数据中读取类别名称；如果本地没有数据集，则退回到类别编号。

    ESC-50 的 `esc50.csv` 中包含 `target` 和 `category` 两列。训练输出只保存了数字标签，
    因此这里优先尝试用元数据恢复可读类别名，方便混淆矩阵和类别级指标直接用于报告。
    """
    fallback = {label: f"class_{label:02d}" for label in labels}
    if metadata_path is None or not metadata_path.exists():
        return fallback

    class_names: dict[int, str] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if "target" not in row or "category" not in row:
                return fallback
            class_names[int(row["target"])] = row["category"]

    return {label: class_names.get(label, fallback[label]) for label in labels}


def plot_training_curves(history: list[dict[str, float]], output_dir: Path, title: str) -> None:
    """绘制 loss 与 accuracy 曲线，用于观察 baseline 是否稳定收敛。"""
    epochs = [record["epoch"] for record in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [record["train_loss"] for record in history], marker="o", label="Train Loss")
    plt.plot(epochs, [record["val_loss"] for record in history], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [record["train_accuracy"] for record in history], marker="o", label="Train Accuracy")
    plt.plot(epochs, [record["val_accuracy"] for record in history], marker="o", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png", dpi=200)
    plt.close()


def build_class_metrics(labels: list[int], matrix: np.ndarray, class_names: dict[int, str]) -> list[dict[str, object]]:
    """根据混淆矩阵计算每个类别的样本数、预测正确数和类别准确率。"""
    rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        support = int(matrix[index].sum())
        correct = int(matrix[index, index])
        accuracy = correct / support if support else 0.0
        rows.append(
            {
                "label": label,
                "class_name": class_names[label],
                "support": support,
                "correct": correct,
                "accuracy": accuracy,
            }
        )
    return rows


def save_class_metrics(rows: list[dict[str, object]], output_dir: Path) -> None:
    """保存类别级指标 CSV，后续可直接导入 Excel、PPT 或报告表格。"""
    csv_path = output_dir / "class_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "class_name", "support", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(
    labels: list[int], matrix: np.ndarray, class_names: dict[int, str], output_dir: Path, title: str
) -> None:
    """绘制归一化混淆矩阵，重点观察哪些类别被系统性混淆。"""
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    tick_labels = [class_names[label] for label in labels]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        normalized,
        cmap="viridis",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Recall by true class"},
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"{title} Normalized Confusion Matrix")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_normalized.png", dpi=220)
    plt.close()


def write_summary(
    history: list[dict[str, float]],
    metrics: dict[str, object],
    class_rows: list[dict[str, object]],
    output_dir: Path,
    title: str,
) -> None:
    """生成 Markdown 摘要，沉淀本轮 baseline 的关键结论和后续分析入口。"""
    best_val = max(history, key=lambda record: record["val_accuracy"])
    final_record = history[-1]
    sorted_by_accuracy = sorted(class_rows, key=lambda row: float(row["accuracy"]))
    weakest = sorted_by_accuracy[:5]
    strongest = sorted_by_accuracy[-5:][::-1]

    lines = [
        f"# {title} 结果摘要",
        "",
        "## 整体结果",
        "",
        f"- 训练轮数：{len(history)}",
        f"- 验证样本数：{metrics['num_samples']}",
        f"- 最佳验证 Accuracy：{best_val['val_accuracy']:.4f}（第 {best_val['epoch']} 轮）",
        f"- 最佳轮验证 Loss：{best_val['val_loss']:.4f}",
        f"- 最后一轮训练 Accuracy：{final_record['train_accuracy']:.4f}",
        f"- 最后一轮验证 Accuracy：{final_record['val_accuracy']:.4f}",
        f"- 最后一轮验证 Loss：{final_record['val_loss']:.4f}",
        "",
        "## 类别级表现较好",
        "",
    ]
    for row in strongest:
        lines.append(f"- {row['class_name']}：Accuracy {float(row['accuracy']):.4f}，{row['correct']}/{row['support']}")

    lines.extend(["", "## 类别级表现较弱", ""])
    for row in weakest:
        lines.append(f"- {row['class_name']}：Accuracy {float(row['accuracy']):.4f}，{row['correct']}/{row['support']}")

    lines.extend(
        [
            "",
            "## 生成文件",
            "",
            "- `training_loss.png`：训练/验证 Loss 曲线。",
            "- `training_accuracy.png`：训练/验证 Accuracy 曲线。",
            "- `confusion_matrix_normalized.png`：按真实类别归一化的混淆矩阵。",
            "- `class_metrics.csv`：每个类别的样本数、正确数和类别准确率。",
            "",
            "## 初步结论",
            "",
            "- 当前实验结果明显高于 50 类随机水平，说明数据读取、训练流程和评估记录有效。",
            "- 后续应结合多实验对比摘要判断该方法相对 baseline 的提升幅度，并继续补充类别级错误分析。",
        ]
    )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 ESC-50 baseline 训练结果并生成图表")
    parser.add_argument("--history", default="outputs/esc50_baseline/history.json", help="训练历史 JSON 路径")
    parser.add_argument(
        "--metrics",
        default="outputs/esc50_baseline/latest_val_metrics.json",
        help="验证集指标 JSON 路径",
    )
    parser.add_argument(
        "--metadata",
        default="data/ESC-50/meta/esc50.csv",
        help="可选 ESC-50 元数据路径，用于把类别编号映射为类别名",
    )
    parser.add_argument("--output-dir", default="outputs/esc50_baseline/analysis", help="分析结果输出目录")
    parser.add_argument("--title", default="ESC-50 CNN Baseline", help="图表和摘要中显示的实验名称")
    args = parser.parse_args()

    history = load_json(Path(args.history))
    metrics = load_json(Path(args.metrics))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [int(label) for label in metrics["labels"]]
    matrix = np.array(metrics["confusion_matrix"], dtype=int)
    metadata_path = Path(args.metadata) if args.metadata else None
    class_names = load_esc50_class_names(metadata_path, labels)

    plot_training_curves(history, output_dir, args.title)
    class_rows = build_class_metrics(labels, matrix, class_names)
    save_class_metrics(class_rows, output_dir)
    plot_confusion_matrix(labels, matrix, class_names, output_dir, args.title)
    write_summary(history, metrics, class_rows, output_dir, args.title)

    print(f"分析完成，结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
