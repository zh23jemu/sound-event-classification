from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.io import wavfile
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Esc50Item:
    """ESC-50 单条样本的元信息。"""

    audio_path: Path
    target: int
    category: str
    fold: int


class ESC50Dataset(Dataset[tuple[torch.Tensor, int]]):
    """ESC-50 数据集读取器。

    目录约定：
        data/ESC-50/
          meta/esc50.csv
          audio/*.wav

    当前实现优先服务 baseline：读取音频、统一采样率、裁剪/填充到固定长度。
    Log-Mel 特征在训练循环中统一计算，便于后续把特征计算迁移到 GPU。
    """

    def __init__(
        self,
        root: str | Path,
        folds: list[int],
        sample_rate: int,
        duration_seconds: float,
    ) -> None:
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_seconds)
        meta_path = self.root / "meta" / "esc50.csv"

        if not meta_path.exists():
            raise FileNotFoundError(
                "没有找到 ESC-50 元数据文件："
                f"{meta_path}\n请先下载 ESC-50 并解压到配置中的 data.root。"
            )

        metadata = pd.read_csv(meta_path)
        metadata = metadata[metadata["fold"].isin(folds)].reset_index(drop=True)
        if metadata.empty:
            raise ValueError(f"指定 fold 没有样本：{folds}")

        self.items = [
            Esc50Item(
                audio_path=self.root / "audio" / row["filename"],
                target=int(row["target"]),
                category=str(row["category"]),
                fold=int(row["fold"]),
            )
            for _, row in metadata.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.items[index]
        waveform, sr = self._load_wav(item.audio_path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = self._fix_length(waveform)
        return waveform, item.target

    @staticmethod
    def _load_wav(path: Path) -> tuple[torch.Tensor, int]:
        """使用 SciPy 读取 WAV，避开新版 torchaudio 对 TorchCodec/FFmpeg 的依赖。

        ESC-50 原始音频是普通 WAV 文件，用 `scipy.io.wavfile` 读取足够稳定。
        这里手动把整数 PCM 归一化到 [-1, 1]，并统一转成 `[channels, samples]`
        形状，保持后续特征提取流程不变。
        """

        sample_rate, audio = wavfile.read(path)
        audio_array = np.asarray(audio)

        if np.issubdtype(audio_array.dtype, np.integer):
            max_value = float(np.iinfo(audio_array.dtype).max)
            audio_array = audio_array.astype(np.float32) / max_value
        else:
            audio_array = audio_array.astype(np.float32)

        if audio_array.ndim == 1:
            audio_array = audio_array[np.newaxis, :]
        else:
            audio_array = audio_array.T

        waveform = torch.from_numpy(np.ascontiguousarray(audio_array))
        return waveform, sample_rate

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """把音频裁剪或补零到固定长度，保证 batch 内张量形状一致。"""

        if waveform.size(-1) > self.num_samples:
            return waveform[..., : self.num_samples]
        if waveform.size(-1) < self.num_samples:
            pad = self.num_samples - waveform.size(-1)
            return torch.nn.functional.pad(waveform, (0, pad))
        return waveform
