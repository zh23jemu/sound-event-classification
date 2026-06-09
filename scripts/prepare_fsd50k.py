from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    """检查 FSD50K 数据目录是否满足训练脚本要求。"""

    parser = argparse.ArgumentParser(description="检查 FSD50K 数据目录结构")
    parser.add_argument("--root", default="data/FSD50K", help="FSD50K 解压后的根目录")
    args = parser.parse_args()

    root = Path(args.root)
    required_paths = [
        root / "FSD50K.dev_audio",
        root / "FSD50K.eval_audio",
        root / "FSD50K.ground_truth" / "dev.csv",
        root / "FSD50K.ground_truth" / "eval.csv",
        root / "FSD50K.ground_truth" / "vocabulary.csv",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("FSD50K 目录缺少必要文件或目录：\n" + "\n".join(str(path) for path in missing))

    dev = pd.read_csv(root / "FSD50K.ground_truth" / "dev.csv")
    eval_set = pd.read_csv(root / "FSD50K.ground_truth" / "eval.csv")
    vocabulary = pd.read_csv(root / "FSD50K.ground_truth" / "vocabulary.csv", header=None)
    train_count = int((dev["split"] == "train").sum())
    val_count = int((dev["split"] == "val").sum())
    eval_count = len(eval_set)

    print(f"FSD50K 数据目录检查通过：{root}")
    print(f"训练样本数：{train_count}")
    print(f"验证样本数：{val_count}")
    print(f"评估样本数：{eval_count}")
    print(f"类别数：{len(vocabulary)}")


if __name__ == "__main__":
    main()
