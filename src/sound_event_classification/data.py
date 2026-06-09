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


@dataclass(frozen=True)
class Fsd50kItem:
    """FSD50K 单条样本的元信息。

    FSD50K 是 clip-level 多标签数据集，一条音频可能同时对应多个声音事件。
    因此这里保存的是多热标签向量，而不是 ESC-50 中的单个整数类别。
    """

    audio_path: Path
    labels: list[str]
    target: torch.Tensor
    split: str


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


class FSD50KDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """FSD50K 多标签数据集读取器。

    目录约定：
        data/FSD50K/
          FSD50K.dev_audio/*.wav
          FSD50K.eval_audio/*.wav
          FSD50K.ground_truth/dev.csv
          FSD50K.ground_truth/eval.csv
          FSD50K.ground_truth/vocabulary.csv

    `dev.csv` 内部包含官方 `train` / `val` 划分，`eval.csv` 用于最终评估。
    当前实现优先支持 `train`、`val` 和 `eval` 三种 split。
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        sample_rate: int,
        duration_seconds: float,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_seconds)
        self.ground_truth_dir = self.root / "FSD50K.ground_truth"
        self.vocabulary_path = self.ground_truth_dir / "vocabulary.csv"
        self.label_to_index, self.index_to_label = self._load_vocabulary(self.vocabulary_path)

        if split in {"train", "val"}:
            metadata_path = self.ground_truth_dir / "dev.csv"
            audio_dir = self.root / "FSD50K.dev_audio"
            metadata = pd.read_csv(metadata_path)
            metadata = metadata[metadata["split"] == split].reset_index(drop=True)
        elif split == "eval":
            metadata_path = self.ground_truth_dir / "eval.csv"
            audio_dir = self.root / "FSD50K.eval_audio"
            metadata = pd.read_csv(metadata_path)
        else:
            raise ValueError("FSD50K split 只能是 train、val 或 eval")

        if metadata.empty:
            raise ValueError(f"FSD50K 指定 split 没有样本：{split}")

        self.items = [
            Fsd50kItem(
                audio_path=audio_dir / f"{row['fname']}.wav",
                labels=self._split_labels(str(row["labels"])),
                target=self._labels_to_multihot(str(row["labels"])),
                split=split,
            )
            for _, row in metadata.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        waveform, sr = ESC50Dataset._load_wav(item.audio_path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = self._fix_length(waveform)
        return waveform, item.target.clone()

    @staticmethod
    def _split_labels(labels: str) -> list[str]:
        """解析 FSD50K CSV 中逗号分隔的多标签字符串。"""

        return [label.strip() for label in labels.split(",") if label.strip()]

    @staticmethod
    def _load_vocabulary(path: Path) -> tuple[dict[str, int], list[str]]:
        """读取 FSD50K vocabulary.csv，建立标签到索引的映射。"""

        if not path.exists():
            raise FileNotFoundError(f"没有找到 FSD50K vocabulary.csv：{path}")

        vocabulary = pd.read_csv(path, header=None)
        if vocabulary.shape[1] < 2:
            raise ValueError(f"FSD50K vocabulary.csv 至少应包含索引和标签两列：{path}")

        label_names = vocabulary.iloc[:, 1].astype(str).tolist()
        label_to_index = {label: index for index, label in enumerate(label_names)}
        return label_to_index, label_names

    def _labels_to_multihot(self, labels: str) -> torch.Tensor:
        """把 FSD50K 标签列表转换为 multi-hot 向量。"""

        target = torch.zeros(len(self.index_to_label), dtype=torch.float32)
        for label in self._split_labels(labels):
            if label in self.label_to_index:
                target[self.label_to_index[label]] = 1.0
        return target

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """把可变长度 FSD50K 音频裁剪或补零到固定长度。"""

        if waveform.size(-1) > self.num_samples:
            return waveform[..., : self.num_samples]
        if waveform.size(-1) < self.num_samples:
            pad = self.num_samples - waveform.size(-1)
            return torch.nn.functional.pad(waveform, (0, pad))
        return waveform
