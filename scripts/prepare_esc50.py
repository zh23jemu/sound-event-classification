from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """检查 ESC-50 数据目录是否符合训练脚本约定。

    该脚本暂时不自动下载数据，避免在不同网络环境下出现不可控失败。用户只需
    从 ESC-50 官方仓库下载并解压，然后用本脚本确认目录结构是否正确。
    """

    parser = argparse.ArgumentParser(description="检查 ESC-50 数据目录结构")
    parser.add_argument("--root", default="data/ESC-50", help="ESC-50 解压后的根目录")
    args = parser.parse_args()

    root = Path(args.root)
    meta = root / "meta" / "esc50.csv"
    audio = root / "audio"

    if meta.exists() and audio.exists():
        wav_count = len(list(audio.glob("*.wav")))
        print(f"ESC-50 数据目录检查通过：{root}")
        print(f"音频文件数量：{wav_count}")
        print(f"元数据文件：{meta}")
        return

    print("ESC-50 数据目录尚未准备好。")
    print(f"期望目录：{root}")
    print("需要包含：")
    print("  meta/esc50.csv")
    print("  audio/*.wav")
    print("建议下载来源：https://github.com/karolpiczak/ESC-50")


if __name__ == "__main__":
    main()

