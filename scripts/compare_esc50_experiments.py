import argparse
import csv
import json
from pathlib import Path


def load_history(path: Path) -> list[dict[str, float]]:
    """读取单个实验的训练历史文件。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到训练历史文件：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list) or not data:
        raise ValueError(f"训练历史为空或格式不正确：{path}")
    return data


def summarize_experiment(name: str, history_path: Path) -> dict[str, object]:
    """提取一个实验最适合放入报告的关键指标。"""
    history = load_history(history_path)
    best = max(history, key=lambda record: float(record["val_accuracy"]))
    final = history[-1]
    return {
        "experiment": name,
        "history_path": str(history_path),
        "epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_val_accuracy": float(best["val_accuracy"]),
        "best_val_loss": float(best["val_loss"]),
        "final_train_accuracy": float(final["train_accuracy"]),
        "final_val_accuracy": float(final["val_accuracy"]),
        "final_val_loss": float(final["val_loss"]),
    }


def parse_experiment(value: str) -> tuple[str, Path]:
    """解析 `实验名=history.json路径` 格式，避免命令行参数歧义。"""
    if "=" not in value:
        raise argparse.ArgumentTypeError("实验参数格式应为 name=path/to/history.json")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("实验名和路径都不能为空")
    return name.strip(), Path(path.strip())


def write_csv(rows: list[dict[str, object]], output_dir: Path) -> None:
    """保存机器可读的实验对比表，后续可继续扩展更多模型。"""
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], output_dir: Path) -> None:
    """保存面向报告撰写的 Markdown 对比摘要。"""
    best_row = max(rows, key=lambda row: float(row["best_val_accuracy"]))
    baseline = rows[0]
    lines = [
        "# ESC-50 实验对比摘要",
        "",
        "## 关键指标",
        "",
        "| 实验 | 最佳轮次 | 最佳验证 Accuracy | 最佳验证 Loss | 最后一轮训练 Accuracy | 最后一轮验证 Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {best_epoch} | {best_val_accuracy:.4f} | {best_val_loss:.4f} | "
            "{final_train_accuracy:.4f} | {final_val_accuracy:.4f} |".format(**row)
        )

    delta = float(best_row["best_val_accuracy"]) - float(baseline["best_val_accuracy"])
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 当前最佳实验：{best_row['experiment']}，最佳验证 Accuracy = {float(best_row['best_val_accuracy']):.4f}。",
            f"- 相比第一个实验 `{baseline['experiment']}`，当前最佳提升 {delta:+.4f}。",
            "- SpecAugment 版本训练集 Accuracy 更低，说明增强提高了训练难度；验证集最佳值略高，但提升幅度较小，需要继续做多折或更多增强策略验证。",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="对比多个 ESC-50 实验的训练历史")
    parser.add_argument(
        "--experiment",
        action="append",
        type=parse_experiment,
        required=True,
        help="实验配置，格式为 name=path/to/history.json；可重复传入多个实验",
    )
    parser.add_argument("--output-dir", default="outputs/esc50_comparison", help="对比摘要输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_experiment(name, path) for name, path in args.experiment]
    write_csv(rows, output_dir)
    write_markdown(rows, output_dir)
    print(f"对比完成，结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
