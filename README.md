# 声音事件分类项目

本项目研究基于深度学习的声音事件分类。当前阶段从文献与计划进入代码实现，目标是先在 ESC-50 上跑通一个可复现 baseline，再比较预训练模型、数据增强和训练策略的影响。

## 当前代码结构

| 路径 | 作用 |
| --- | --- |
| `configs/esc50_baseline.yaml` | ESC-50 baseline 默认配置 |
| `scripts/prepare_esc50.py` | 检查 ESC-50 数据目录是否准备好 |
| `scripts/train.py` | 训练和验证入口 |
| `src/sound_event_classification/data.py` | ESC-50 数据读取与音频裁剪/填充 |
| `src/sound_event_classification/features.py` | Log-Mel Spectrogram 特征提取 |
| `src/sound_event_classification/models.py` | 当前轻量 CNN baseline，后续可接入 AST/ViT |
| `src/sound_event_classification/metrics.py` | 单标签分类指标 |
| `slurm/train_esc50_baseline.sbatch` | Slurm 集群训练脚本 |
| `项目日志.md` | 项目日志模板和实验记录 |

## 环境准备

项目要求使用本地虚拟环境 `.venv`，不要直接使用系统 Python。

Windows 本地安装依赖示例：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux / Slurm 环境安装依赖示例：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

如需在 GPU 集群使用 PyTorch，优先安装 CUDA 12.x 兼容 wheel，例如：

```bash
.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
```

## 数据准备

当前 baseline 使用 ESC-50。请先下载并解压 ESC-50，使目录结构类似：

```text
data/ESC-50/
  meta/esc50.csv
  audio/*.wav
```

检查数据目录：

```powershell
.venv\Scripts\python.exe scripts\prepare_esc50.py --root data\ESC-50
```

## 运行 baseline

Windows 本地运行：

```powershell
.venv\Scripts\python.exe scripts\train.py --config configs\esc50_baseline.yaml
```

Linux / Slurm 集群运行：

```bash
sbatch slurm/train_esc50_baseline.sbatch
```

如需临时覆盖 Slurm 分区，例如切换到 GPU 分区：

```bash
sbatch --partition=gpu slurm/train_esc50_baseline.sbatch
```

## 输出说明

默认输出目录为 `outputs/esc50_baseline`，包括：

- `history.json`：每轮训练和验证指标。
- `latest_val_metrics.json`：最近一轮验证集指标和混淆矩阵。
- `best_model.pt`：验证 Accuracy 最好的模型权重。

当前项目不会整体忽略 `outputs/`，便于保留小型结果文件用于报告、截图和分析；但大模型权重和 checkpoint 文件默认不建议提交。
