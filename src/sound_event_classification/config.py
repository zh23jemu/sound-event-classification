from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件，并返回普通字典。

    参数：
        path: 配置文件路径。

    返回：
        配置字典。训练脚本会继续对其中的路径和默认值做解释。
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式不正确，应为 YAML 字典：{config_path}")
    return data

