#!/usr/bin/env bash
set -euo pipefail

device="3,4"
dataset_name="nlphuji/flickr30k"
dataset_cache=""
clip_model="openai/clip-vit-base-patch32"
model_type="weighted_sum_cls"
checkpoint="checkpoints/weighted_sum_cls/best.pt"
layers=(3 6 9 12)
alpha=0.3
split="test"
index=0
seed=42
output_dir="visualization/weighted_sum_cls"

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
  --output_dir "$output_dir" \
  "$@" 2>&1 | tee "$log_dir/run.log"
