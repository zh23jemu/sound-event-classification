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


class SpecAugment(torch.nn.Module):
    """对 Log-Mel Spectrogram 做 SpecAugment 数据增强。

    SpecAugment 的核心思想是在频率轴和时间轴上随机遮挡若干连续区域，让模型不要
    过度依赖某一小段频率或时间模式。这里仅在训练阶段启用，验证阶段必须关闭，
    这样 baseline 与增强实验的评估口径保持一致。
    """

    def __init__(
        self,
        time_mask_param: int,
        freq_mask_param: int,
        num_time_masks: int,
        num_freq_masks: int,
        p: float,
    ) -> None:
        super().__init__()
        self.time_mask_param = max(int(time_mask_param), 0)
        self.freq_mask_param = max(int(freq_mask_param), 0)
        self.num_time_masks = max(int(num_time_masks), 0)
        self.num_freq_masks = max(int(num_freq_masks), 0)
        self.p = float(p)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """返回增强后的频谱图。

        参数：
            spec: 形状为 `[batch, channels, n_mels, frames]` 的归一化 Log-Mel 特征。

        返回：
            与输入形状相同的张量。被遮挡区域填 0，因为当前特征已经按样本归一化，
            0 近似表示该样本的平均能量位置。
        """

        if not self.training or self.p <= 0 or torch.rand((), device=spec.device).item() > self.p:
            return spec

        augmented = spec.clone()
        batch_size, _, num_freqs, num_frames = augmented.shape

        # 频率遮挡：对每个样本随机选取若干频带置零，模拟部分频率信息缺失。
        for batch_index in range(batch_size):
            for _ in range(self.num_freq_masks):
                width = self._sample_width(self.freq_mask_param, num_freqs, augmented.device)
                if width == 0:
                    continue
                start = self._sample_start(num_freqs, width, augmented.device)
                augmented[batch_index, :, start : start + width, :] = 0.0

            # 时间遮挡：对每个样本随机遮挡若干连续时间帧，提高对局部事件缺失的鲁棒性。
            for _ in range(self.num_time_masks):
                width = self._sample_width(self.time_mask_param, num_frames, augmented.device)
                if width == 0:
                    continue
                start = self._sample_start(num_frames, width, augmented.device)
                augmented[batch_index, :, :, start : start + width] = 0.0

        return augmented

    @staticmethod
    def _sample_width(mask_param: int, axis_size: int, device: torch.device) -> int:
        """随机采样遮挡宽度，保证不会超过当前特征轴长度。"""

        max_width = min(mask_param, axis_size)
        if max_width <= 0:
            return 0
        return int(torch.randint(0, max_width + 1, (1,), device=device).item())

    @staticmethod
    def _sample_start(axis_size: int, width: int, device: torch.device) -> int:
        """随机采样遮挡起点，保证 `[start, start + width)` 不越界。"""

        if width >= axis_size:
            return 0
        return int(torch.randint(0, axis_size - width + 1, (1,), device=device).item())
