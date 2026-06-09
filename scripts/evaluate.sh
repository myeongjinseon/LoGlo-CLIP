#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${CONDA_SH:-/mnt/sdd/trans4/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-loglo_clip}"

mkdir -p results logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" python models/evaluate.py \
    --checkpoint checkpoints/weighted_sum_main/best.pt \
    --model_type weighted_sum \
    --layers 3 6 9 12 \
    "$@" 2>&1 | tee "logs/evaluate_${TIMESTAMP}.log"
