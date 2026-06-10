#!/usr/bin/env bash
set -euo pipefail

device="3,4"
dataset_name="nlphuji/flickr30k"
dataset_cache=""
clip_model="openai/clip-vit-base-patch32"
model_type="weighted_sum_cls"
checkpoint="checkpoints/weighted_sum_cls/best.pt"
split="test"
layers=(3 6 9 12)
batch_size=64
alpha=0.3
seed=42
num_workers=4

tag="evaluate_${model_type}"
save_dir="checkpoints/${tag}"
result_dir="results/${tag}"
log_dir="logs/${tag}"

mkdir -p "$save_dir" "$result_dir" "$log_dir"
cp "${BASH_SOURCE[0]}" "$save_dir/run.sh"
export CUDA_VISIBLE_DEVICES="$device"
export TOKENIZERS_PARALLELISM=false

cache_args=()
[[ -n "$dataset_cache" ]] && cache_args=(--dataset_cache "$dataset_cache")
python experiments/evaluate_retrieval.py \
  --model_type "$model_type" \
  --checkpoint "$checkpoint" \
  --split "$split" \
  --dataset_name "$dataset_name" \
  "${cache_args[@]}" \
  --clip_model "$clip_model" \
  --layers "${layers[@]}" \
  --batch_size "$batch_size" \
  --alpha "$alpha" \
  --seed "$seed" \
  --num_workers "$num_workers" \
  --result_dir "$result_dir" \
  "$@" 2>&1 | tee "$log_dir/run.log"
