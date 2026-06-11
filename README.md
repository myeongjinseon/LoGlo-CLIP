# LoGlo-CLIP: Local-to-Global Layer Fusion for CLIP Image-Text Retrieval

## Student name and Youtube Link
Myeongjin Seon 22200376
https://youtu.be/J05XfoPAiKg

## Overview

LoGlo-CLIP is a lightweight visual encoder adapter for CLIP ViT-B/32. It
combines CLS tokens from multiple vision-transformer layers with a learned
static weighted sum and evaluates the resulting representation on Flickr30k
image-text retrieval.

The repository includes standard retrieval evaluation, hard-negative
retrieval, corruption robustness, fusion and layer ablations, a CLS versus
patch-mean ablation, and feature visualization.

## Method

The main model, **Static WeightedSumCLS**, extracts CLS features from CLIP
vision layers `[3, 6, 9, 12]`. Learned scalar logits are normalized with a
softmax:

```text
w_i = softmax(a)_i
h_fused = sum_i w_i h_i^CLS
z_fusion = normalize(W h_fused)
z_image = normalize(z_CLIP + alpha * z_fusion)
```

The CLIP image and text encoders remain frozen. Only the layer weights,
fusion module, and projection layer are trained.

## Hyperparameter Management

All experiment hyperparameters are defined in `scripts/*.sh`.
Each script creates a checkpoint/result/log directory and copies itself into
the checkpoint directory for reproducibility.

## Installation

```bash
conda create -n loglo_clip python=3.9
conda activate loglo_clip
pip install -r requirements.txt
```

## Dataset

Experiments use [`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k).
Images and annotations are loaded through the HuggingFace `datasets` cache.
Set `dataset_cache` near the top of a shell script to use a custom cache path.

## Training

```bash
bash scripts/train_weighted_sum.sh
```

The main run writes `best.pt`, `last.pt`, retrieval metrics, and learned layer
weights below its checkpoint and result directories.

## Evaluation

```bash
bash scripts/evaluate_retrieval.sh
```

Reported metrics include Recall@1/5/10, MRR, median rank, mean rank, and
nDCG@10 for image-to-text and text-to-image retrieval.

## Ablation Studies

```bash
bash scripts/ablation_fusion.sh
bash scripts/ablation_layers.sh
bash scripts/ablation_patch_mean.sh
```

The fusion study compares frozen CLIP, Linear(L12), static WeightedSum-CLS,
self-attention, cross-attention, and static WeightedSum-PatchMean.

## Analysis

```bash
bash scripts/hard_negative.sh
bash scripts/robustness.sh
bash scripts/visualize_features.sh
```

## Results

## Results

| Experiment              | Main metric                    | Result                                                                                               |
| ----------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Flickr30k retrieval     | I2T / T2I Recall@1             | CLIP: 70.50 / 68.10 → LoGlo-CLIP: 74.90 / 75.00                                                      |
| Hard-negative retrieval | Top-1 accuracy / MRR           | CLIP: 87.20 / 0.9293 → LoGlo-CLIP: 88.20 / 0.9344                                                    |
| Corruption robustness   | I2T Recall@1 under corruptions | Higher absolute R@1 than CLIP in most corruptions; smaller R@1 drop on colour jitter and centre crop |


Generated CSV files are stored under `results/`.

## Project Structure

```text
loglo/         reusable data, model, loss, metric, engine, and figure code
experiments/   argparse-based Python experiment entry points
scripts/       explicit hyperparameters and reproducible shell commands
checkpoints/   model checkpoints and copied run scripts
results/       CSV results
visualization/ paper-ready figures
logs/          command output
```

## Citation

```bibtex
@article{logloclip2026,
  title   = {LoGlo-CLIP: Local-to-Global Layer Fusion for CLIP Image-Text Retrieval},
  author  = {Myeongjin Seon},
  journal = {Manuscript},
  year    = {2026}
}
```

## Acknowledgement

This project builds on OpenAI CLIP, HuggingFace Transformers and Datasets,
and the Flickr30k dataset.
