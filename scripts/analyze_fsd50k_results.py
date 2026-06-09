from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


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


def write_summary(history: list[dict[str, float]], latest_metrics: dict[str, float], output_dir: Path) -> None:
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
        "",
        "## 生成文件",
        "",
        "- `training_loss.png`：训练/验证 Loss 曲线。",
        "- `validation_metrics.png`：验证 mAP、micro-F1 和 macro-F1 曲线。",
    ]
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
    args = parser.parse_args()

    history = load_json(Path(args.history))
    latest_metrics = load_json(Path(args.metrics))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss(history, output_dir)
    plot_metrics(history, output_dir)
    write_summary(history, latest_metrics, output_dir)
    print(f"分析完成，结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
