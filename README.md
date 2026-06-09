# 声音事件分类项目

本项目研究基于深度学习的声音事件分类。当前阶段从文献与计划进入代码实现，目标是先在 ESC-50 上跑通一个可复现 baseline，再比较预训练模型、数据增强和训练策略的影响。

## 当前代码结构

| 路径 | 作用 |
| --- | --- |
| `configs/esc50_baseline.yaml` | ESC-50 baseline 默认配置 |
| `scripts/prepare_esc50.py` | 检查 ESC-50 数据目录是否准备好 |
| `scripts/prepare_fsd50k.py` | 检查 FSD50K 数据目录是否准备好 |
| `scripts/train.py` | 训练和验证入口 |
| `scripts/train_ast.py` | 预训练 AST 微调入口 |
| `scripts/train_fsd50k_ast.py` | FSD50K 预训练 AST 多标签微调入口 |
| `scripts/evaluate_fsd50k_ast.py` | 读取 FSD50K 最佳模型并导出验证集逐样本预测概率 |
| `scripts/analyze_esc50_results.py` | 分析训练结果，生成曲线、混淆矩阵和类别级指标 |
| `scripts/analyze_fsd50k_results.py` | 分析 FSD50K 多标签结果，生成 mAP/F1 曲线、类别级指标、阈值敏感性和摘要 |
| `src/sound_event_classification/data.py` | ESC-50 数据读取与音频裁剪/填充 |
| `src/sound_event_classification/features.py` | Log-Mel Spectrogram 特征提取 |
| `src/sound_event_classification/models.py` | 当前轻量 CNN baseline，后续可接入 AST/ViT |
| `src/sound_event_classification/metrics.py` | 单标签分类指标 |
| `slurm/train_esc50_baseline.sbatch` | Slurm 集群训练脚本 |
| `slurm/train_esc50_ast.sbatch` | Slurm GPU AST 微调脚本 |
| `slurm/train_fsd50k_ast.sbatch` | Slurm GPU FSD50K 多标签微调脚本 |
| `项目日志.md` | 项目日志模板和实验记录 |
| `实验结果分析.md` | 可放入最终报告的实验结果分析章节草稿 |
| `模型报告.md` | 整合文献背景、方法、实验和结论的模型报告初稿 |
| `metadata/esc50.csv` | ESC-50 官方元数据，用于把类别编号映射为真实类别名 |

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

FSD50K 扩展实验需要下载官方 Zenodo 数据。由于音频文件较大，建议放在服务器共享存储或容量充足的位置，再在项目 `data/` 下放软链接。目录最终应类似：

```text
data/FSD50K/
  FSD50K.dev_audio/
  FSD50K.eval_audio/
  FSD50K.ground_truth/
    dev.csv
    eval.csv
    vocabulary.csv
```

服务器下载和解压示例：

```bash
mkdir -p data/FSD50K_downloads data/FSD50K
cd data/FSD50K_downloads

wget -O FSD50K.ground_truth.zip "https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip?download=1"
wget -O FSD50K.metadata.zip "https://zenodo.org/record/4060432/files/FSD50K.metadata.zip?download=1"

wget -O FSD50K.dev_audio.z01 "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01?download=1"
wget -O FSD50K.dev_audio.z02 "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02?download=1"
wget -O FSD50K.dev_audio.z03 "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03?download=1"
wget -O FSD50K.dev_audio.z04 "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04?download=1"
wget -O FSD50K.dev_audio.z05 "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05?download=1"
wget -O FSD50K.dev_audio.zip "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip?download=1"

wget -O FSD50K.eval_audio.z01 "https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01?download=1"
wget -O FSD50K.eval_audio.zip "https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip?download=1"

zip -s 0 FSD50K.dev_audio.zip --out FSD50K.dev_audio.unsplit.zip
zip -s 0 FSD50K.eval_audio.zip --out FSD50K.eval_audio.unsplit.zip

unzip FSD50K.ground_truth.zip -d ../FSD50K
unzip FSD50K.metadata.zip -d ../FSD50K
unzip FSD50K.dev_audio.unsplit.zip -d ../FSD50K
unzip FSD50K.eval_audio.unsplit.zip -d ../FSD50K

cd ../..
.venv/bin/python scripts/prepare_fsd50k.py --root data/FSD50K
```

## 运行 baseline

Windows 本地运行：

```powershell
.venv\Scripts\python.exe scripts\train.py --config configs\esc50_baseline.yaml
```

运行 CNN + SpecAugment 对比实验：

```powershell
.venv\Scripts\python.exe scripts\train.py --config configs\esc50_cnn_specaugment.yaml
```

运行预训练 AST 微调实验：

```powershell
.venv\Scripts\python.exe scripts\train_ast.py --config configs\esc50_ast.yaml
```

Linux / Slurm 集群运行：

```bash
sbatch slurm/train_esc50_baseline.sbatch
```

如需临时覆盖 Slurm 分区，例如切换到 GPU 分区：

```bash
sbatch --partition=gpu slurm/train_esc50_baseline.sbatch
```

如需在 Slurm 上运行 SpecAugment 配置：

```bash
sbatch --partition=ISP --export=ALL,CONFIG=configs/esc50_cnn_specaugment.yaml slurm/train_esc50_baseline.sbatch
```

AST 微调建议使用 GPU 分区：

```bash
sbatch slurm/train_esc50_ast.sbatch
```

FSD50K 多标签 AST 微调建议使用 GPU 分区：

```bash
sbatch slurm/train_fsd50k_ast.sbatch
```

如需使用 8 小时时限的 `gpuHz` 分区，可提交：

```bash
sbatch --partition=gpuHz slurm/train_fsd50k_ast.sbatch
```

首次运行 AST 前，如果服务器 `.venv` 还没有 Transformers 依赖，可先安装：

```bash
.venv/bin/python -m pip install transformers accelerate
```

## 输出说明

默认输出目录为 `outputs/esc50_baseline`，包括：

- `history.json`：每轮训练和验证指标。
- `latest_val_metrics.json`：最近一轮验证集指标和混淆矩阵。
- `best_model.pt`：验证 Accuracy 最好的模型权重。

SpecAugment 对比实验默认输出到 `outputs/esc50_cnn_specaugment`，目录结构与 baseline 保持一致。

AST 微调实验默认输出到 `outputs/esc50_ast`，同样会保存 `history.json`、`latest_val_metrics.json` 和 `best_model.pt`。

FSD50K 多标签 AST 实验默认输出到 `outputs/fsd50k_ast`，主要指标为 `mAP`、`micro_f1` 和 `macro_f1`。

当前 FSD50K 初步结果：

- 最佳验证 mAP：`0.6208`
- 最佳轮验证 micro-F1：`0.6955`
- 最佳轮验证 macro-F1：`0.4810`

当前项目不会整体忽略 `outputs/`，便于保留小型结果文件用于报告、截图和分析；但大模型权重和 checkpoint 文件默认不建议提交。

## 结果分析

训练完成并同步 `history.json`、`latest_val_metrics.json` 后，可运行：

```powershell
.venv\Scripts\python.exe scripts\analyze_esc50_results.py
```

脚本默认生成到 `outputs/esc50_baseline/analysis`：

- `training_loss.png`：训练/验证 Loss 曲线。
- `training_accuracy.png`：训练/验证 Accuracy 曲线。
- `confusion_matrix_normalized.png`：归一化混淆矩阵。
- `class_metrics.csv`：每个类别的样本数、正确数和类别准确率。
- `summary.md`：适合写入报告的结果摘要。

项目已保存 ESC-50 官方元数据 `metadata/esc50.csv`。如需显式指定真实类别名映射，可运行：

```powershell
.venv\Scripts\python.exe scripts\analyze_esc50_results.py `
  --metadata metadata/esc50.csv
```

多个实验完成后，可生成对比摘要：

```powershell
.venv\Scripts\python.exe scripts\compare_esc50_experiments.py `
  --experiment "CNN baseline=outputs/esc50_baseline/history.json" `
  --experiment "CNN + SpecAugment=outputs/esc50_cnn_specaugment/history.json"
```

对比结果默认输出到 `outputs/esc50_comparison`。

FSD50K 整体曲线、类别级指标和阈值敏感性分析可运行：

```powershell
.venv\Scripts\python.exe scripts\analyze_fsd50k_results.py
```

如果只同步了 `history.json` 和 `latest_val_metrics.json`，脚本会先生成整体曲线和摘要。若要进一步生成 `class_metrics.csv`、`threshold_sensitivity.csv` 和 `threshold_sensitivity.png`，需要服务器上已有 `outputs/fsd50k_ast/best_model.pt`，并先导出验证集逐样本预测概率：

```bash
.venv/bin/python scripts/evaluate_fsd50k_ast.py \
  --config configs/fsd50k_ast.yaml \
  --checkpoint outputs/fsd50k_ast/best_model.pt \
  --output outputs/fsd50k_ast/val_predictions.json

.venv/bin/python scripts/analyze_fsd50k_results.py
```

服务器生成后建议只提交小型分析文件，不提交 `best_model.pt`：

```bash
git add outputs/fsd50k_ast/val_predictions.json \
        outputs/fsd50k_ast/analysis/class_metrics.csv \
        outputs/fsd50k_ast/analysis/threshold_sensitivity.csv \
        outputs/fsd50k_ast/analysis/threshold_sensitivity.png \
        outputs/fsd50k_ast/analysis/summary.md

git commit -m "exp: 补充 FSD50K 类别级和阈值分析" \
  -m "导出 FSD50K 验证集逐样本预测概率，生成类别级 AP/F1 和阈值敏感性分析，用于补充多标签结果讨论。"
git push
```

## 常见问题

### torchaudio / torchcodec 解码报错

部分新版 `torchaudio.load()` 会通过 TorchCodec/FFmpeg 解码音频，容易遇到集群运行库不匹配问题。当前代码已经改为用 `scipy.io.wavfile` 读取 ESC-50 的 WAV 文件，因此不需要安装 `torchcodec`。

如果服务器已经装过 `torchcodec`，可以保留；训练脚本不会再调用它。更新代码后请重新运行 `git pull`，再提交 Slurm 任务。
