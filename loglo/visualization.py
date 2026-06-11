"""Visualization helpers for CLIP layers and learned fusion weights."""

import math
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

from .utils import ensure_dir, unwrap_model


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def normalize_map(values):
    values = values.float()
    return (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)


def _image_array(image):
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(
            "image must be a PIL image or an HxW RGB/RGBA numpy array."
        )
    return array


def _heatmap_array(heatmap):
    if torch.is_tensor(heatmap):
        heatmap = heatmap.detach().float().cpu().numpy()
    heatmap = np.asarray(heatmap, dtype=np.float32).squeeze()
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a single-channel 2D array.")
    return heatmap


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
    heatmap = torch.as_tensor(_heatmap_array(heatmap))
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


def resize_signed_heatmap(heatmap, size):
    heatmap = torch.as_tensor(_heatmap_array(heatmap))
    return (
        F.interpolate(
            heatmap[None, None],
            size=size,
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        .cpu()
        .numpy()
    )


def save_heatmap_overlay(image, heatmap, path, title, alpha=0.40, dpi=300):
    plt = _pyplot()
    path = Path(path)
    ensure_dir(path.parent)
    array = _image_array(image)
    resized = resize_heatmap(heatmap, (array.shape[0], array.shape[1]))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(array)
    axis.imshow(resized, cmap="inferno", alpha=alpha, vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_difference_overlay(image, difference, path, title, alpha=0.55, dpi=300):
    plt = _pyplot()
    path = Path(path)
    ensure_dir(path.parent)
    array = _image_array(image)
    resized = resize_signed_heatmap(
        difference, (array.shape[0], array.shape[1])
    )
    limit = max(float(np.abs(resized).max()), 1e-8)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(array)
    artist = axis.imshow(
        resized,
        cmap="coolwarm",
        alpha=alpha,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_title(title)
    axis.axis("off")
    colorbar = figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.035)
    colorbar.set_label("Fusion gain over CLIP")
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_original_image(image, path):
    path = Path(path)
    ensure_dir(path.parent)
    if isinstance(image, Image.Image):
        image.save(path)
    else:
        Image.fromarray(_image_array(image).astype(np.uint8)).save(path)


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


def _patch_grid(values):
    patch_count = values.size(-1)
    grid = math.isqrt(patch_count)
    if grid * grid != patch_count:
        raise ValueError(f"Cannot form a square grid from {patch_count} patches.")
    return values.reshape(grid, grid)


def postprocess_relevance(heatmap, gamma=2.0, threshold=0.20):
    """Normalize, suppress weak responses, and sharpen a relevance map."""

    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if not 0 <= threshold < 1:
        raise ValueError("threshold must be in [0, 1).")
    heatmap = normalize_map(heatmap)
    heatmap = torch.where(
        heatmap >= threshold,
        (heatmap - threshold) / (1.0 - threshold),
        torch.zeros_like(heatmap),
    )
    return heatmap.clamp(0, 1).pow(gamma)


def _caption_relevance(
    patch_embeddings,
    text_embedding,
    gamma=2.0,
    threshold=0.20,
):
    patch_embeddings = F.normalize(patch_embeddings, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)
    relevance = torch.einsum("bpd,bd->bp", patch_embeddings, text_embedding)[0]
    relevance = relevance.clamp_min(0)
    return postprocess_relevance(
        _patch_grid(relevance),
        gamma=gamma,
        threshold=threshold,
    )


def _clip_cls_patch_relevance(module, outputs, temperature=0.07):
    """Compare final projected patch tokens with CLIP's projected CLS token."""

    final_tokens = module.clip.vision_model.post_layernorm(
        outputs.last_hidden_state
    )
    projected_tokens = F.normalize(
        module.clip.visual_projection(final_tokens),
        dim=-1,
    )
    patch_alignment = (
        projected_tokens[:, 1:] * projected_tokens[:, :1]
    ).sum(dim=-1)
    relevance = torch.softmax(patch_alignment / temperature, dim=-1)[0]
    return normalize_map(_patch_grid(relevance))


def _cross_attention_relevance(module, outputs):
    """Recover the learned CLS-to-patch cross-attention map."""

    source_states = [
        outputs.hidden_states[layer] for layer in module.layers[:-1]
    ]
    query = outputs.hidden_states[module.layers[-1]][:, :1]
    context = torch.cat(source_states, dim=1)
    _, attention = module.attention(
        query=query,
        key=context,
        value=context,
        need_weights=True,
        average_attn_weights=True,
    )

    sequence_length = source_states[0].size(1)
    per_layer_patches = []
    for offset in range(len(source_states)):
        start = offset * sequence_length
        per_layer_patches.append(
            attention[0, 0, start + 1 : start + sequence_length]
        )
    relevance = torch.stack(per_layer_patches).mean(dim=0)
    return normalize_map(_patch_grid(relevance))


def _cross_attention_caption_relevance(
    module,
    outputs,
    text_embedding,
    gamma=2.0,
    threshold=0.20,
):
    """Localize learned cross-attention routes that also match the caption."""

    source_layers = module.layers[:-1]
    query = outputs.hidden_states[module.layers[-1]][:, :1]
    source_states = [outputs.hidden_states[layer] for layer in source_layers]
    context = torch.cat(source_states, dim=1)
    _, attention = module.attention(
        query=query,
        key=context,
        value=context,
        need_weights=True,
        average_attn_weights=True,
    )

    sequence_length = source_states[0].size(1)
    per_layer_scores = []
    for offset, hidden_state in enumerate(source_states):
        start = offset * sequence_length
        patch_attention = attention[:, 0, start + 1 : start + sequence_length]
        patch_embeddings = F.normalize(
            module.projector(hidden_state[:, 1:]),
            dim=-1,
        )
        caption_scores = torch.einsum(
            "bpd,bd->bp",
            patch_embeddings,
            F.normalize(text_embedding, dim=-1),
        ).clamp_min(0)
        per_layer_scores.append(patch_attention * caption_scores)

    relevance = torch.stack(per_layer_scores, dim=1).sum(dim=1)[0]
    return postprocess_relevance(
        _patch_grid(relevance),
        gamma=gamma,
        threshold=threshold,
    )


def _draw_visualization_panel(
    axis,
    visualization,
    original_image,
    overlay_alpha,
):
    if torch.is_tensor(visualization):
        array = visualization.detach().float().cpu().numpy()
    else:
        array = np.asarray(visualization)
    array = np.asarray(array).squeeze()
    if array.ndim == 2:
        image_array = _image_array(original_image)
        resized = resize_heatmap(array, image_array.shape[:2])
        axis.imshow(image_array)
        axis.imshow(
            resized,
            cmap="inferno",
            alpha=overlay_alpha,
            vmin=0,
            vmax=1,
        )
        return
    axis.imshow(_image_array(visualization))


def save_comparison_grid(
    original_image,
    layer3_visualization,
    layer6_visualization,
    layer9_visualization,
    layer12_visualization,
    clip_final_visualization,
    fusion_visualization,
    rollout_visualization,
    caption,
    output_path,
    overlay_alpha=0.40,
    dpi=300,
    grid_mode="legacy_attention",
):
    """Save an eight-panel CLIP final versus fusion comparison figure."""

    plt = _pyplot()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if grid_mode == "legacy_attention":
        titles = (
            "Original\nInput image",
            "Layer3\nRaw CLS attention",
            "Layer6\nRaw CLS attention",
            "Layer9\nRaw CLS attention",
            "Layer12\nLast-block attention",
            "CLIP Final\nProjected CLS-patch relevance",
            "Fusion\nCross-scale attention",
            "Rollout\nCumulative attention path",
        )
        figure_title = "Multi-Scale Fusion vs. CLIP Final Representation"
    elif grid_mode == "caption_relevance":
        titles = (
            "Original\nInput image",
            "Layer 3\nCLS Relevance",
            "Layer 6\nCLS Relevance",
            "Layer 9\nCLS Relevance",
            "Layer 12\nCLS Relevance",
            "CLIP Final\nFinal Representation",
            "LoGlo Fusion\nWeightedSum Relevance",
            "Rollout\nCumulative attention path",
        )
        figure_title = (
            "LoGlo-CLIP Weighted Fusion vs. CLIP Final Representation"
        )
    else:
        raise ValueError(f"Unknown grid mode: {grid_mode}")
    visualizations = (
        original_image,
        layer3_visualization,
        layer6_visualization,
        layer9_visualization,
        layer12_visualization,
        clip_final_visualization,
        fusion_visualization,
        rollout_visualization,
    )

    figure, axes = plt.subplots(2, 4, figsize=(16, 9.2))
    for index, (axis, title, visualization) in enumerate(
        zip(axes.flat, titles, visualizations)
    ):
        if index == 0:
            axis.imshow(_image_array(original_image))
        else:
            _draw_visualization_panel(
                axis,
                visualization,
                original_image,
                overlay_alpha,
            )
        axis.set_title(title, fontsize=13, fontweight="bold", pad=10)
        axis.axis("off")

    figure.suptitle(
        figure_title,
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    wrapped_caption = textwrap.fill(
        f'"{caption or "No caption provided"}"', width=125
    )
    figure.text(
        0.5,
        0.925,
        f"Caption:\n{wrapped_caption}",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.025,
        right=0.90,
        bottom=0.035,
        top=0.83,
        wspace=0.055,
        hspace=0.20,
    )

    relevance_mappable = ScalarMappable(
        norm=Normalize(vmin=0, vmax=1), cmap="inferno"
    )
    relevance_mappable.set_array([])
    colorbar_axis = figure.add_axes((0.925, 0.17, 0.014, 0.58))
    colorbar = figure.colorbar(relevance_mappable, cax=colorbar_axis)
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.set_label("Normalized relevance", fontsize=12, fontweight="bold")
    colorbar.ax.tick_params(labelsize=10)

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


save_visualization_grid = save_comparison_grid


@torch.no_grad()
def extract_visual_maps(
    model,
    pixel_values,
    layers,
    input_ids=None,
    attention_mask=None,
    relevance_gamma=2.0,
    relevance_threshold=0.20,
):
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
    maps["clip_final_cls_patch"] = _clip_cls_patch_relevance(
        module,
        outputs,
    )
    if module.model_type == "cross_attention":
        maps["cross_attention"] = _cross_attention_relevance(
            module,
            outputs,
        )

    if input_ids is not None and attention_mask is not None:
        text_embedding = module.encode_text(input_ids, attention_mask)
        for layer in layers:
            hidden_state = outputs.hidden_states[layer]
            projected_tokens = module.clip.visual_projection(hidden_state)
            layer_query = F.normalize(
                F.normalize(projected_tokens[:, 0], dim=-1) + text_embedding,
                dim=-1,
            )
            maps[f"layer_{layer}_cls_relevance"] = _caption_relevance(
                projected_tokens[:, 1:],
                layer_query,
                gamma=relevance_gamma,
                threshold=relevance_threshold,
            )
        final_tokens = module.clip.vision_model.post_layernorm(
            outputs.last_hidden_state
        )
        clip_patch_embeddings = module.clip.visual_projection(
            final_tokens[:, 1:]
        )
        maps["clip_final"] = _caption_relevance(
            clip_patch_embeddings,
            text_embedding,
            gamma=relevance_gamma,
            threshold=relevance_threshold,
        )

        if tuple(layers) != tuple(module.layers):
            raise ValueError(
                "Visualization layers must match the fusion model's layers."
            )
        if hasattr(module, "normalized_weights"):
            layer_patch_tokens = torch.stack(
                [outputs.hidden_states[layer][:, 1:] for layer in layers],
                dim=1,
            )
            weights = module.normalized_weights()
            fused_patch_tokens = (
                layer_patch_tokens * weights[None, :, None, None]
            ).sum(dim=1)
            fusion_patch_embeddings = F.normalize(
                module.projector(fused_patch_tokens), dim=-1
            )
            maps["loglo_fusion"] = _caption_relevance(
                fusion_patch_embeddings,
                text_embedding,
                gamma=relevance_gamma,
                threshold=relevance_threshold,
            )

            # Keep the retrieval residual as a separate diagnostic map.
            residual_patch_embeddings = F.normalize(
                F.normalize(clip_patch_embeddings, dim=-1)
                + module.alpha * fusion_patch_embeddings,
                dim=-1,
            )
            maps["loglo_residual"] = _caption_relevance(
                residual_patch_embeddings,
                text_embedding,
                gamma=relevance_gamma,
                threshold=relevance_threshold,
            )
        elif module.model_type == "cross_attention":
            maps["loglo_fusion"] = _cross_attention_caption_relevance(
                module,
                outputs,
                text_embedding,
                gamma=relevance_gamma,
                threshold=relevance_threshold,
            )
        else:
            raise TypeError(
                "Visualization supports Static WeightedSum or Cross-Attention "
                "fusion checkpoints."
            )
        maps["difference"] = maps["loglo_fusion"] - maps["clip_final"]
    return maps
