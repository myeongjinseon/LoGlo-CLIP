#!/usr/bin/env bash
set -euo pipefail

device="3,4"
dataset_name="nlphuji/flickr30k"
dataset_cache=""
clip_model="openai/clip-vit-base-patch32"
model_types=("clip" "weighted_sum_cls")
checkpoint="checkpoints/weighted_sum_cls/best.pt"
linear_checkpoint="checkpoints/ablation_fusion/linear_l12/best.pt"
patch_checkpoint="checkpoints/weighted_sum_patch_mean/best.pt"
layers=(3 6 9 12)
alpha=0.3
split="test"
num_samples=1000
num_negatives=5
batch_size=64
seed=42
num_workers=4

tag="hard_negative"
save_dir="checkpoints/${tag}"
result_dir="results/${tag}"
log_dir="logs/${tag}"
mkdir -p "$save_dir" "$result_dir" "$log_dir"
cp "${BASH_SOURCE[0]}" "$save_dir/run.sh"
export CUDA_VISIBLE_DEVICES="$device"
export TOKENIZERS_PARALLELISM=false

cache_args=()
[[ -n "$dataset_cache" ]] && cache_args=(--dataset_cache "$dataset_cache")
python experiments/hard_negative.py \
  --model_types "${model_types[@]}" \
  --checkpoint "$checkpoint" \
  --linear_checkpoint "$linear_checkpoint" \
  --patch_checkpoint "$patch_checkpoint" \
  --dataset_name "$dataset_name" \
  "${cache_args[@]}" \
  --clip_model "$clip_model" \
  --layers "${layers[@]}" \
  --alpha "$alpha" \
  --split "$split" \
  --num_samples "$num_samples" \
  --num_negatives "$num_negatives" \
  --batch_size "$batch_size" \
  --seed "$seed" \
  --num_workers "$num_workers" \
  --result_dir "$result_dir" \
  "$@" 2>&1 | tee "$log_dir/run.log"
