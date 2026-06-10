#!/usr/bin/env bash
set -euo pipefail

device="3,4"
dataset_name="nlphuji/flickr30k"
dataset_cache=""
clip_model="openai/clip-vit-base-patch32"
model_types=("weighted_sum_cls" "weighted_sum_patch_mean")
layers=(3 6 9 12)
batch_size=64
epochs=5
lr=1e-4
weight_decay=0.01
alpha=0.3
seed=42
num_workers=4

tag="ablation_patch_mean"
result_dir="results/${tag}"
log_dir="logs/${tag}"
mkdir -p "$result_dir" "$log_dir"
export CUDA_VISIBLE_DEVICES="$device"
export TOKENIZERS_PARALLELISM=false

cache_args=()
[[ -n "$dataset_cache" ]] && cache_args=(--dataset_cache "$dataset_cache")
for model_type in "${model_types[@]}"; do
  save_dir="checkpoints/${tag}/${model_type}"
  mkdir -p "$save_dir"
  cp "${BASH_SOURCE[0]}" "$save_dir/run.sh"
  python experiments/ablation_patch_mean.py \
    --experiment_name "ablation_patch_mean" \
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
    "$@" 2>&1 | tee "$log_dir/${model_type}.log"
done
