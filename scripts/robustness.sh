#!/usr/bin/env bash
set -euo pipefail

device="3,4"
dataset_name="nlphuji/flickr30k"
dataset_cache=""
clip_model="openai/clip-vit-base-patch32"
model_types=("clip" "weighted_sum_cls" "weighted_sum_patch_mean")
checkpoint_weighted="checkpoints/weighted_sum_cls/best.pt"
checkpoint_patch="checkpoints/weighted_sum_patch_mean/best.pt"
corruptions=("original" "gaussian_blur" "grayscale" "color_jitter" "center_crop" "occlusion")
layers=(3 6 9 12)
alpha=0.3
split="test"
num_samples=1000
batch_size=64
seed=42
num_workers=4

tag="robustness"
save_dir="checkpoints/${tag}"
result_dir="results/${tag}"
log_dir="logs/${tag}"
mkdir -p "$save_dir" "$result_dir" "$log_dir"
cp "${BASH_SOURCE[0]}" "$save_dir/run.sh"
export CUDA_VISIBLE_DEVICES="$device"
export TOKENIZERS_PARALLELISM=false

cache_args=()
[[ -n "$dataset_cache" ]] && cache_args=(--dataset_cache "$dataset_cache")
python experiments/robustness.py \
  --model_types "${model_types[@]}" \
  --checkpoint_weighted "$checkpoint_weighted" \
  --checkpoint_patch "$checkpoint_patch" \
  --corruptions "${corruptions[@]}" \
  --dataset_name "$dataset_name" \
  "${cache_args[@]}" \
  --clip_model "$clip_model" \
  --layers "${layers[@]}" \
  --alpha "$alpha" \
  --split "$split" \
  --num_samples "$num_samples" \
  --batch_size "$batch_size" \
  --seed "$seed" \
  --num_workers "$num_workers" \
  --result_dir "$result_dir" \
  "$@" 2>&1 | tee "$log_dir/run.log"
