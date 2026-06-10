"""Retrieval, hard-negative, and robustness metrics."""

import numpy as np
import torch


def _ranking_metrics(similarity):
    count = similarity.size(0)
    targets = torch.arange(count, device=similarity.device)
    order = similarity.argsort(dim=1, descending=True)
    ranks = order.eq(targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    ranks_float = ranks.float()
    metrics = {
        "r1": ranks.le(1).float().mean().item() * 100,
        "r5": ranks.le(5).float().mean().item() * 100,
        "r10": ranks.le(10).float().mean().item() * 100,
        "mrr": ranks_float.reciprocal().mean().item(),
        "median_rank": ranks_float.median().item(),
        "mean_rank": ranks_float.mean().item(),
    }
    ideal = torch.ones(count, device=similarity.device)
    gains = ranks.le(10).float() / torch.log2(ranks_float + 1)
    metrics["ndcg10"] = (gains / ideal).mean().item()
    return metrics


def retrieval_metrics(image_embeddings, text_embeddings):
    if image_embeddings.size(0) != text_embeddings.size(0):
        raise ValueError("Retrieval metrics require aligned image-text pairs.")
    similarity = image_embeddings @ text_embeddings.T
    return {
        "i2t": _ranking_metrics(similarity),
        "t2i": _ranking_metrics(similarity.T),
    }


def flatten_retrieval_metrics(metrics, prefix=""):
    row = {}
    for direction in ("i2t", "t2i"):
        for name, value in metrics[direction].items():
            row[f"{prefix}{direction}_{name}"] = value
    return row


def hard_negative_metrics(ranks, margins):
    ranks = torch.as_tensor(ranks, dtype=torch.float)
    return {
        "top1_accuracy": ranks.eq(1).float().mean().item() * 100,
        "mrr": ranks.reciprocal().mean().item(),
        "mean_rank": ranks.mean().item(),
        "median_rank": ranks.median().item(),
        "mean_margin": float(np.mean(margins)),
    }


def robustness_summary(rows, model_key="model", corruption_key="corruption"):
    summaries = []
    by_model = {}
    for row in rows:
        by_model.setdefault(row[model_key], []).append(row)
    for model, model_rows in by_model.items():
        original = next(
            row for row in model_rows if row[corruption_key] == "original"
        )
        for row in model_rows:
            summary = dict(row)
            summary["i2t_r1_drop"] = original["i2t_r1"] - row["i2t_r1"]
            summary["i2t_r5_drop"] = original["i2t_r5"] - row["i2t_r5"]
            summaries.append(summary)
    return summaries
