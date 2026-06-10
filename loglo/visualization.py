"""Visualization helpers for CLIP layers and learned fusion weights."""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .utils import ensure_dir, unwrap_model


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def normalize_map(values):
    values = values.float()
    return (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)


def attention_to_heatmap(attention):
    patch_count = attention.size(-1) - 1
    grid = math.isqrt(patch_count)
    if grid * grid != patch_count:
        raise ValueError(f"Cannot form a square grid from {patch_count} patches.")
    values = attention[0].mean(dim=0)[0, 1:]
    return normalize_map(values.reshape(grid, grid))


def feature_to_heatmap(hidden_state):
    patches = hidden_state[0, 1:]
    values = patches.norm(dim=-1)
    grid = math.isqrt(values.numel())
    if grid * grid != values.numel():
        raise ValueError(f"Cannot form a square grid from {values.numel()} patches.")
    return normalize_map(values.reshape(grid, grid))


def attention_rollout(attentions):
    token_count = attentions[0].size(-1)
    identity = torch.eye(token_count, device=attentions[0].device)
    joint = identity
    for attention in attentions:
        averaged = attention[0].mean(dim=0) + identity
        averaged = averaged / averaged.sum(dim=-1, keepdim=True)
        joint = averaged @ joint
    patch_count = token_count - 1
    grid = math.isqrt(patch_count)
    return normalize_map(joint[0, 1:].reshape(grid, grid))


def resize_heatmap(heatmap, size):
    return (
        F.interpolate(
            heatmap[None, None],
            size=size,
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        .clamp(0, 1)
        .cpu()
        .numpy()
    )


def save_heatmap_overlay(image, heatmap, path, title, alpha=0.48, dpi=300):
    plt = _pyplot()
    path = Path(path)
    ensure_dir(path.parent)
    array = np.asarray(image)
    resized = resize_heatmap(heatmap, (array.shape[0], array.shape[1]))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(array)
    axis.imshow(resized, cmap="inferno", alpha=alpha, vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_original_image(image, path):
    path = Path(path)
    ensure_dir(path.parent)
    image.save(path)


def save_fusion_weights(model, path, dpi=300):
    module = unwrap_model(model)
    if not hasattr(module, "normalized_weights"):
        raise TypeError("This model does not expose learned fusion weights.")
    weights = module.normalized_weights().detach().cpu().numpy()
    plt = _pyplot()
    path = Path(path)
    ensure_dir(path.parent)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.bar([str(layer) for layer in module.layers], weights, color="#31688e")
    axis.set_xlabel("CLIP vision layer")
    axis.set_ylabel("Softmax weight")
    axis.set_ylim(0, max(1.0, float(weights.max()) * 1.15))
    axis.set_title("Learned Static Fusion Weights")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


@torch.no_grad()
def extract_visual_maps(model, pixel_values, layers):
    module = unwrap_model(model)
    outputs = module.clip.vision_model(
        pixel_values=pixel_values,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    maps = {}
    for layer in layers:
        attention_index = layer - 1
        if outputs.attentions and attention_index < len(outputs.attentions):
            maps[f"layer_{layer}_attention"] = attention_to_heatmap(
                outputs.attentions[attention_index]
            )
        maps[f"layer_{layer}_feature"] = feature_to_heatmap(
            outputs.hidden_states[layer]
        )
    if outputs.attentions:
        maps["rollout"] = attention_rollout(outputs.attentions)
    return maps
