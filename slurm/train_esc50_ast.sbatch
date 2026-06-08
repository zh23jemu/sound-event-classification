#!/bin/bash
# ESC-50 预训练 AST 微调脚本，仅用于 Linux Slurm GPU 环境。
# 该实验是项目主线的 Transformer/ViT 对比项，资源需求明显高于 CNN baseline。

#SBATCH --job-name=esc50-ast
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/slurm

# 使用项目本地虚拟环境，避免调用系统 Python。
# 首次运行前如果缺少 transformers，请在登录节点执行：
#   .venv/bin/python -m pip install transformers accelerate
CONFIG="${CONFIG:-configs/esc50_ast.yaml}"
.venv/bin/python scripts/train_ast.py --config "${CONFIG}"
