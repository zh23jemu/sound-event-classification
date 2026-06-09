#!/bin/bash
# FSD50K 预训练 AST 多标签微调脚本，仅用于 Linux Slurm GPU 环境。
# FSD50K 规模明显大于 ESC-50，建议先用 5 epoch 验证流程和指标，再决定是否加长训练。

#SBATCH --job-name=fsd50k-ast
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/slurm

CONFIG="${CONFIG:-configs/fsd50k_ast.yaml}"
.venv/bin/python scripts/train_fsd50k_ast.py --config "${CONFIG}"
