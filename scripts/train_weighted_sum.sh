#!/usr/bin/env bash
set -euo pipefail

device="3,4"
dataset_name="nlphuji/flickr30k"
dataset_cache=""
clip_model="openai/clip-vit-base-patch32"
model_type="weighted_sum_cls"
layers=(3 6 9 12)
batch_size=64
epochs=5
lr=1e-4
weight_decay=0.01
alpha=0.3
seed=42
num_workers=4

tag="weighted_sum_cls"
save_dir="checkpoints/${tag}"
result_dir="results/${tag}"
log_dir="logs/${tag}"

mkdir -p "$save_dir" "$result_dir" "$log_dir" visualization
cp "${BASH_SOURCE[0]}" "$save_dir/run.sh"
export CUDA_VISIBLE_DEVICES="$device"
export TOKENIZERS_PARALLELISM=false

cache_args=()
[[ -n "$dataset_cache" ]] && cache_args=(--dataset_cache "$dataset_cache")
python experiments/train_weighted_sum.py \
  --model_type "$model_type" \
  --dataset_name "$dataset_name" \
  "${cache_args[@]}" \
  --clip_model "$clip_model" \
  --layers "${layers[@]}" \
  --batch_size "$batch_size" \
  --epochs "$epochs" \
  --lr "$lr" \
  --weight_decay "$weight_decay" \
  --alpha "$alpha" \
  --seed "$seed" \
  --num_workers "$num_workers" \
  --save_dir "$save_dir" \
  --result_dir "$result_dir" \
  "$@" 2>&1 | tee "$log_dir/run.log"
