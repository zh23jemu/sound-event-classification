from __future__ import annotations

import torch
from torch import nn


class CnnBaseline(nn.Module):
    """用于 ESC-50 快速验证的轻量 CNN baseline。

    该模型不是最终创新点，而是用来验证数据读取、Log-Mel 特征、训练循环和
    评价指标是否全部可运行。后续接入 AST/ViT 时，可以沿用相同训练脚本。
    """

    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        """构建一个卷积、归一化、激活和下采样模块。"""

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int, dropout: float) -> nn.Module:
    """按名称创建模型。

    当前只实现 `cnn_baseline`。后续如果接入 AST/ViT，可以在这里扩展模型名称，
    从而不改训练脚本主体。
    """

    if name == "cnn_baseline":
        return CnnBaseline(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"未知模型名称：{name}")

