from __future__ import annotations

import torch
import torchaudio


class LogMelSpectrogram(torch.nn.Module):
    """将波形转换为固定形状的 Log-Mel Spectrogram。

    该模块只负责特征转换，不负责读文件和裁剪。把它写成 `torch.nn.Module`
    的好处是后续可以自然放进训练流程，也方便在 GPU 上加速特征计算。
    """

    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """返回归一化后的 Log-Mel 特征。

        参数：
            waveform: 形状为 `[channels, samples]` 或 `[batch, channels, samples]` 的音频。

        返回：
            若输入是单条音频，返回 `[1, n_mels, frames]`；若输入是 batch，
            返回 `[batch, 1, n_mels, frames]`，可直接送入 CNN。
        """

        squeeze_batch = waveform.dim() == 2
        if squeeze_batch:
            waveform = waveform.unsqueeze(0)

        # 多声道音频转为单声道，避免不同数据源的通道数影响模型输入。
        if waveform.size(1) > 1:
            waveform = waveform.mean(dim=1, keepdim=True)

        spec = self.to_db(self.mel(waveform))
        spec = (spec - spec.mean(dim=(-2, -1), keepdim=True)) / (
            spec.std(dim=(-2, -1), keepdim=True) + 1e-6
        )

        return spec.squeeze(0) if squeeze_batch else spec

