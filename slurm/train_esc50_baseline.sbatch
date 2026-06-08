#!/bin/bash
# ESC-50 baseline 训练脚本，仅用于 Linux Slurm 集群环境。
# 默认使用 CPU 分区 defq，适合先验证数据读取、Log-Mel 特征和轻量 CNN baseline。
# 如果后续改成 AST/ViT GPU 微调，可在提交时用：
#   sbatch --partition=gpu slurm/train_esc50_baseline.sbatch

#SBATCH --job-name=esc50-baseline
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/slurm

# 使用集群上的项目本地虚拟环境，避免调用系统 Python。
# 如果集群环境还没有 .venv，请先在登录节点按 requirements.txt 创建并安装依赖。
# 可通过环境变量覆盖配置文件，例如：
#   sbatch --partition=ISP --export=ALL,CONFIG=configs/esc50_cnn_specaugment.yaml slurm/train_esc50_baseline.sbatch
CONFIG="${CONFIG:-configs/esc50_baseline.yaml}"
.venv/bin/python scripts/train.py --config "${CONFIG}"
