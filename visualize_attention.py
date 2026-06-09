import argparse
import math
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from dataset import DEFAULT_ANNOTATIONS_PATH, load_annotations, resolve_image_path
from model import ScaleWiseFusionModule, load_fusion_checkpoint
from models.common import checkpoint_config, load_checkpoint, load_clip
from models.train import WeightedSumFusion


MODEL_NAME = "openai/clip-vit-base-patch32"
SELECTED_LAYERS = (3, 6, 9, 12)
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize CLIP and fusion attention on a local image."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_fusion.pt"),
    )
    parser.add_argument("--image_path", type=Path)
    parser.add_argument(
        "--image_name",
        help="Image filename from annotations, for example 000000_1007129816.jpg.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based sample index within --split.",
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--output_dir", type=Path, default=Path("visualization"))
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.48)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc
    return plt


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(name)


def load_clip_model(model_name, device):
    try:
        model = CLIPModel.from_pretrained(
            model_name,
            attn_implementation="eager",
        )
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise
        model = CLIPModel.from_pretrained(model_name)
    return model.to(device)


def select_local_sample(args):
    if args.image_path is not None and args.image_name is not None:
        raise ValueError("Use either --image_path or --image_name, not both.")

    if args.image_path is not None:
        image_path = args.image_path.expanduser()
        if not image_path.is_absolute():
            image_path = (PROJECT_ROOT / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        caption = ""
        try:
            records = load_annotations(args.annotations)
            target = image_path.resolve()
            for record in records:
                candidate = resolve_image_path(
                    record["image_path"],
                    args.annotations,
                )
                if candidate.resolve() == target:
                    caption = record["caption"]
                    break
        except FileNotFoundError:
            pass
        return image_path, caption

    records = load_annotations(args.annotations, args.split)
    if not args.image_name:
        if args.index < 0 or args.index >= len(records):
            raise IndexError(
                f"--index must be between 0 and {len(records) - 1}."
            )
        record = records[args.index]
        return (
            resolve_image_path(record["image_path"], args.annotations),
            record["caption"],
        )
    matching_records = [
        record
        for record in records
        if record["image_filename"] == args.image_name
        or Path(record["image_path"]).name == args.image_name
    ]
    if not matching_records:
        raise FileNotFoundError(
            f"Image {args.image_name!r} was not found in split "
            f"{args.split!r} of {args.annotations}."
        )
    if len(matching_records) > 1:
        raise ValueError(
            f"Image name {args.image_name!r} is not unique in split "
            f"{args.split!r}. Use --image_path instead."
        )
    record = matching_records[0]
    return (
        resolve_image_path(record["image_path"], args.annotations),
        record["caption"],
    )


def normalize_map(attention_map):
    attention_map = attention_map.float()
    minimum = attention_map.min()
    maximum = attention_map.max()
    return (attention_map - minimum) / (maximum - minimum).clamp_min(1e-8)


def normalize_distribution(values):
    values = values.float().clamp_min(0)
    return values / values.sum().clamp_min(1e-8)


def infer_patch_grid(num_patch_tokens):
    grid_size = math.isqrt(num_patch_tokens)
    if grid_size * grid_size != num_patch_tokens:
        raise ValueError(
            f"Expected a square patch grid, found {num_patch_tokens} tokens."
        )
    return grid_size


def extract_cls_attention(attention, grid_size):
    patches = attention[0].mean(dim=0)[0, 1:]
    return normalize_distribution(patches).reshape(grid_size, grid_size)


def extract_fusion_attention(attention, sequence_length, grid_size):
    # The CLS query attends to concatenated layer 3/6/9 token sequences.
    cls_attention = attention[0, 0]
    per_layer_patches = []
    for layer_offset in range(3):
        start = layer_offset * sequence_length
        per_layer_patches.append(
            cls_attention[start + 1 : start + sequence_length]
        )
    patches = torch.stack(per_layer_patches).mean(dim=0)
    return normalize_distribution(patches).reshape(grid_size, grid_size)


def compute_clip_final_attention(
    vision_outputs,
    vision_model,
    visual_projection,
    grid_size,
):
    # CLIP has no attention block after layer 12. To visualize the actual
    # retrieval representation, compare every final patch token with the
    # projected CLS token used by CLIP's image embedding.
    final_tokens = vision_model.post_layernorm(
        vision_outputs.last_hidden_state
    )
    projected_tokens = F.normalize(
        visual_projection(final_tokens),
        dim=-1,
    )
    projected_cls = projected_tokens[:, :1]
    patch_alignment = (
        projected_tokens[:, 1:] * projected_cls
    ).sum(dim=-1)
    patch_relevance = torch.softmax(patch_alignment / 0.07, dim=-1)[0]
    return patch_relevance.reshape(grid_size, grid_size)


def compute_attention_rollout(attentions, grid_size):
    num_tokens = attentions[0].size(-1)
    joint = torch.eye(
        num_tokens,
        device=attentions[0].device,
        dtype=attentions[0].dtype,
    )
    identity = torch.eye(
        num_tokens,
        device=attentions[0].device,
        dtype=attentions[0].dtype,
    )
    for attention in attentions:
        mean_attention = attention[0].mean(dim=0) + identity
        mean_attention = mean_attention / mean_attention.sum(
            dim=-1,
            keepdim=True,
        )
        joint = mean_attention @ joint
    return normalize_map(joint[0, 1:].reshape(grid_size, grid_size))


def recover_model_input(pixel_values, processor):
    image_processor = processor.image_processor
    mean = torch.tensor(
        image_processor.image_mean,
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    ).view(3, 1, 1)
    std = torch.tensor(
        image_processor.image_std,
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    ).view(3, 1, 1)
    image = pixel_values[0] * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def resize_heatmap(heatmap, height, width):
    resized = F.interpolate(
        heatmap[None, None],
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    return normalize_map(resized).cpu().numpy()


def resize_signed_map(difference_map, height, width):
    return F.interpolate(
        difference_map[None, None],
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )[0, 0].cpu().numpy()


def save_attention_figure(
    plt,
    image,
    heatmap,
    title,
    output_path,
    alpha,
    dpi,
):
    figure, axis = plt.subplots(figsize=(6.4, 6.0))
    axis.imshow(image)
    artist = axis.imshow(
        heatmap,
        cmap="inferno",
        alpha=alpha,
        vmin=0,
        vmax=1,
    )
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.035)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_difference_figure(
    plt,
    image,
    difference_map,
    output_path,
    alpha,
    dpi,
):
    limit = max(float(np.abs(difference_map).max()), 1e-8)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    axis.imshow(image)
    artist = axis.imshow(
        difference_map,
        cmap="coolwarm",
        alpha=alpha,
        vmin=-limit,
        vmax=limit,
    )
    axis.set_title(
        "Fusion - CLIP Final\n"
        "Red: more Fusion emphasis | Blue: more CLIP emphasis",
        fontsize=13,
        fontweight="semibold",
    )
    axis.axis("off")
    colorbar = figure.colorbar(
        artist,
        ax=axis,
        fraction=0.046,
        pad=0.035,
    )
    colorbar.set_label("Signed attention difference", fontsize=10)
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def save_comparison_grid(
    plt,
    image,
    heatmaps,
    caption,
    output_path,
    alpha,
    dpi,
):
    labels = (
        "Original",
        "Layer3",
        "Layer6",
        "Layer9",
        "Layer12",
        "CLIP Final",
        "Fusion",
        "Rollout",
    )
    subtitles = {
        "Original": "Input image",
        "Layer3": "Raw CLS attention",
        "Layer6": "Raw CLS attention",
        "Layer9": "Raw CLS attention",
        "Layer12": "Last-block attention",
        "CLIP Final": "Projected CLS-patch relevance",
        "Fusion": "Cross-scale attention",
        "Rollout": "Cumulative attention path",
    }
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16, 9.2),
        constrained_layout=True,
    )
    artist = None
    for axis, label in zip(axes.flat, labels):
        axis.imshow(image)
        heatmap = heatmaps.get(label)
        if heatmap is not None:
            artist = axis.imshow(
                heatmap,
                cmap="inferno",
                alpha=alpha,
                vmin=0,
                vmax=1,
            )
        axis.set_title(
            f"{label}\n{subtitles[label]}",
            fontsize=11,
            fontweight="semibold",
        )
        axis.axis("off")

    if artist is not None:
        colorbar = figure.colorbar(
            artist,
            ax=axes,
            fraction=0.018,
            pad=0.012,
        )
        colorbar.set_label("Normalized relevance", fontsize=10)
    title = "Multi-Scale Fusion vs. CLIP Final Representation"
    if caption:
        title += "\nCaption:\n\"" + "\n".join(
            textwrap.wrap(caption, width=100)
        ) + "\""
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


@torch.no_grad()
def generate_visualizations(args):
    if not 0 <= args.overlay_alpha <= 1:
        raise ValueError("--overlay_alpha must be between 0 and 1.")

    plt = import_matplotlib()
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    image_path, caption = select_local_sample(args)
    output_dir = output_dir / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    metadata = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint_config(metadata)
    model_name = config.get("model_name", MODEL_NAME)
    layer_indices = config.get(
        "layers",
        config.get("layer_indices", list(SELECTED_LAYERS)),
    )
    if list(layer_indices) != list(SELECTED_LAYERS):
        raise ValueError(
            f"Expected layers {list(SELECTED_LAYERS)}, found {layer_indices}."
        )

    model, processor = load_clip(
        model_name,
        device,
        eager_attention=True,
    )
    model.eval()
    module_kwargs = {
        "embed_dim": model.config.vision_config.hidden_size,
        "projection_dim": model.config.projection_dim,
    }
    model_type = config.get("model_type", "legacy_cross_attention")
    state_keys = set(
        metadata.get(
            "model_state_dict",
            metadata.get("head_state_dict", {}),
        )
    )
    if (
        model_type == "weighted_sum"
        or "model_state_dict" in metadata
        or "layer_logits" in state_keys
    ):
        fusion_original = WeightedSumFusion(
            num_layers=len(SELECTED_LAYERS),
            **module_kwargs,
        ).to(device)
        load_checkpoint(
            checkpoint_path,
            fusion_original,
            map_location=device,
        )
        layer_weights = fusion_original.normalized_weights()
        print(
            "Learned layer weights: "
            + ", ".join(
                f"L{layer}={weight:.4f}"
                for layer, weight in zip(SELECTED_LAYERS, layer_weights)
            )
        )
    else:
        fusion_original = ScaleWiseFusionModule(
            pooling="cls",
            **module_kwargs,
        ).to(device)
        fusion_mean = ScaleWiseFusionModule(
            pooling="mean",
            **module_kwargs,
        ).to(device)
        load_fusion_checkpoint(
            checkpoint_path,
            fusion_original,
            fusion_mean,
            map_location=device,
        )
        layer_weights = None
    fusion_original.eval()

    with Image.open(image_path) as image_file:
        input_image = image_file.convert("RGB")
    pixel_values = processor(
        images=input_image,
        return_tensors="pt",
    )["pixel_values"].to(device)
    outputs = model.vision_model(
        pixel_values=pixel_values,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )
    if not outputs.attentions:
        raise RuntimeError("The model returned no attention tensors.")

    selected_hidden_states = [
        outputs.hidden_states[layer]
        for layer in SELECTED_LAYERS
    ]
    fusion_attention = None
    if layer_weights is None:
        _, fusion_attention = fusion_original(
            *selected_hidden_states,
            return_attention=True,
        )
    num_patch_tokens = outputs.attentions[0].size(-1) - 1
    grid_size = infer_patch_grid(num_patch_tokens)
    model_input = recover_model_input(pixel_values, processor)
    height, width = model_input.shape[:2]

    heatmaps = {"Original": None}
    for layer in SELECTED_LAYERS:
        layer_map = extract_cls_attention(
            outputs.attentions[layer - 1],
            grid_size,
        )
        heatmaps[f"Layer{layer}"] = resize_heatmap(
            layer_map,
            height,
            width,
        )

    if layer_weights is None:
        fusion_map = extract_fusion_attention(
            fusion_attention,
            selected_hidden_states[0].size(1),
            grid_size,
        )
    else:
        fusion_map = sum(
            weight
            * extract_cls_attention(
                outputs.attentions[layer - 1],
                grid_size,
            )
            for layer, weight in zip(SELECTED_LAYERS, layer_weights)
        )
        fusion_map = normalize_distribution(fusion_map)
    heatmaps["Fusion"] = resize_heatmap(
        fusion_map,
        height,
        width,
    )
    clip_final_map = compute_clip_final_attention(
        outputs,
        model.vision_model,
        model.visual_projection,
        grid_size,
    )
    heatmaps["CLIP Final"] = resize_heatmap(
        clip_final_map,
        height,
        width,
    )
    difference_map = resize_signed_map(
        fusion_map - clip_final_map,
        height,
        width,
    )
    rollout_map = compute_attention_rollout(outputs.attentions, grid_size)
    heatmaps["Rollout"] = resize_heatmap(
        rollout_map,
        height,
        width,
    )

    Image.fromarray(np.round(model_input * 255).astype(np.uint8)).save(
        output_dir / "original_image.jpg",
        quality=95,
    )
    for layer in SELECTED_LAYERS:
        layer_description = (
            "Last-Block CLS Attention"
            if layer == SELECTED_LAYERS[-1]
            else "Raw CLS Attention"
        )
        save_attention_figure(
            plt,
            model_input,
            heatmaps[f"Layer{layer}"],
            f"Layer{layer}: {layer_description}",
            output_dir / f"layer{layer}_attention.jpg",
            args.overlay_alpha,
            args.dpi,
        )
    save_attention_figure(
        plt,
        model_input,
        heatmaps["CLIP Final"],
        "CLIP Final Representation Relevance\n"
        "(Projected CLS-Patch Alignment)",
        output_dir / "clip_final_attention.jpg",
        args.overlay_alpha,
        args.dpi,
    )
    save_attention_figure(
        plt,
        model_input,
        heatmaps["Fusion"],
        (
            "Weighted Layer Attention "
            + ", ".join(
                f"L{layer}:{weight:.3f}"
                for layer, weight in zip(SELECTED_LAYERS, layer_weights)
            )
            if layer_weights is not None
            else "Fusion Cross-Scale Attention"
        ),
        output_dir / "fusion_attention.jpg",
        args.overlay_alpha,
        args.dpi,
    )
    save_difference_figure(
        plt,
        model_input,
        difference_map,
        output_dir / "fusion_vs_clip_difference.jpg",
        args.overlay_alpha,
        args.dpi,
    )
    save_attention_figure(
        plt,
        model_input,
        heatmaps["Rollout"],
        "Attention Rollout",
        output_dir / "attention_rollout.jpg",
        args.overlay_alpha,
        args.dpi,
    )
    save_comparison_grid(
        plt,
        model_input,
        heatmaps,
        caption,
        output_dir / "comparison_grid.jpg",
        args.overlay_alpha,
        args.dpi,
    )
    print(f"Image: {image_path}")
    if caption:
        print(f"Caption: {caption}")
    print(
        "Definitions | Layer12: last-block raw CLS attention | "
        "CLIP Final: projected CLS-patch relevance | "
        "Rollout: cumulative attention path"
    )
    print(f"Saved attention visualizations to: {output_dir.resolve()}")


def main():
    generate_visualizations(parse_args())


if __name__ == "__main__":
    main()
