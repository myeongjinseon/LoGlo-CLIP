#!/usr/bin/env bash
set -euo pipefail

source "${CONDA_SH:-/mnt/sdd/trans4/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-loglo_clip}"

device="3,4"
dataset_name="nlphuji/flickr30k"
dataset_cache=""
clip_model="openai/clip-vit-base-patch32"
model_type="cross_attention"
checkpoint="checkpoints/weighted_sum_cls/best.pt"
layers=(3 6 9 12)
alpha=0.3
split="test"
index=5
seed=42
relevance_gamma=2.0
relevance_threshold=0.20
overlay_alpha=0.48
output_dir="visualization/cross_attention"
save_grid=true
grid_output="visualization/loglo_vs_clip_grid.png"
grid_mode="legacy_attention"

tag="visualize_features"
save_dir="checkpoints/${tag}"
log_dir="logs/${tag}"
mkdir -p "$save_dir" "$output_dir" "$log_dir"
cp "${BASH_SOURCE[0]}" "$save_dir/run.sh"
export CUDA_VISIBLE_DEVICES="$device"
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/loglo_clip_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

cache_args=()
[[ -n "$dataset_cache" ]] && cache_args=(--dataset_cache "$dataset_cache")
grid_args=()
[[ "$save_grid" == true ]] && grid_args+=(--save_grid --grid_output "$grid_output")
python experiments/visualize_features.py \
  --model_type "$model_type" \
  --checkpoint "$checkpoint" \
  --dataset_name "$dataset_name" \
  "${cache_args[@]}" \
  --clip_model "$clip_model" \
  --layers "${layers[@]}" \
  --alpha "$alpha" \
  --split "$split" \
  --index "$index" \
  --seed "$seed" \
  --relevance_gamma "$relevance_gamma" \
  --relevance_threshold "$relevance_threshold" \
  --overlay_alpha "$overlay_alpha" \
  --grid_mode "$grid_mode" \
  --output_dir "$output_dir" \
  "${grid_args[@]}" \
  "$@" 2>&1 | tee "$log_dir/run.log"
