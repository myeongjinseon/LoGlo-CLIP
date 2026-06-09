#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${CONDA_SH:-/mnt/sdd/trans4/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-loglo_clip}"

mkdir -p visualization logs
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/loglo_clip_matplotlib}"
mkdir -p "$MPLCONFIGDIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"
INDEX="${1:-0}"

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" python visualize_attention.py \
    --checkpoint checkpoints/weighted_sum_main/best.pt \
    --split test \
    --index "$INDEX" \
    --output_dir visualization \
    2>&1 | tee "logs/visualization_${TIMESTAMP}.log"
