"""Create paper-ready CLIP layer, rollout, and fusion-weight figures."""

import argparse
import sys
from pathlib import Path

from PIL import Image
from transformers import CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loglo.data import Flickr30kDataset, load_flickr30k, select_split
from loglo.engine import load_checkpoint
from loglo.models import build_model
from loglo.utils import ensure_dir, get_device, set_seed
from loglo.visualization import (
    extract_visual_maps,
    save_comparison_grid,
    save_fusion_weights,
    save_heatmap_overlay,
    save_difference_overlay,
    save_original_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="weighted_sum_cls")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_name", default="nlphuji/flickr30k")
    parser.add_argument("--dataset_cache", type=Path)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--layers", type=int, nargs="+", default=(3, 6, 9, 12))
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image_path", type=Path)
    parser.add_argument(
        "--caption",
        default="",
        help="Caption used with --image_path. Dataset samples use their own caption.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("visualization/weighted_sum_cls"))
    parser.add_argument(
        "--save_grid",
        action="store_true",
        help="Save the 2x4 LoGlo-versus-CLIP comparison grid.",
    )
    parser.add_argument(
        "--grid_output",
        type=Path,
        default=Path("visualization/loglo_vs_clip_grid.png"),
    )
    parser.add_argument(
        "--grid_mode",
        choices=("legacy_attention", "caption_relevance"),
        default="legacy_attention",
        help=(
            "legacy_attention reproduces the raw-attention comparison figure; "
            "caption_relevance uses text-conditioned maps."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--relevance_gamma",
        type=float,
        default=2.0,
        help="Gamma correction applied after relevance normalization.",
    )
    parser.add_argument(
        "--relevance_threshold",
        type=float,
        default=0.20,
        help="Normalized activations below this value are suppressed.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.40,
        help="Heatmap overlay opacity.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def select_sample(args):
    if args.image_path:
        if not args.image_path.is_file():
            raise FileNotFoundError(f"Image not found: {args.image_path}")
        with Image.open(args.image_path) as image:
            return image.convert("RGB"), args.caption
    raw = select_split(load_flickr30k(args.dataset_name, args.dataset_cache), args.split)
    dataset = Flickr30kDataset(raw, args.split)
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"--index must be between 0 and {len(dataset) - 1}.")
    sample = dataset[args.index]
    return sample["image"], sample["caption"]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.output_dir)
    if args.model_type not in {"weighted_sum_cls", "cross_attention"}:
        raise ValueError(
            "Feature visualization supports --model_type weighted_sum_cls "
            "or cross_attention."
        )
    image, caption = select_sample(args)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = build_model(
        args.model_type, args.clip_model, tuple(args.layers), args.alpha
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    encoded = processor(
        images=image,
        text=[caption],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    maps = extract_visual_maps(
        model,
        encoded["pixel_values"],
        args.layers,
        encoded["input_ids"],
        encoded["attention_mask"],
        relevance_gamma=args.relevance_gamma,
        relevance_threshold=args.relevance_threshold,
    )
    save_original_image(image, args.output_dir / "original.png")
    for name, heatmap in maps.items():
        if name == "difference":
            save_difference_overlay(
                image,
                heatmap,
                args.output_dir / f"{name}.png",
                "Difference: LoGlo Fusion - CLIP Final",
                dpi=args.dpi,
            )
        else:
            save_heatmap_overlay(
                image,
                heatmap,
                args.output_dir / f"{name}.png",
                name.replace("_", " ").title(),
                alpha=args.overlay_alpha,
                dpi=args.dpi,
            )
    if hasattr(model, "normalized_weights"):
        save_fusion_weights(model, args.output_dir / "fusion_weights.png", args.dpi)
    if args.save_grid:
        if args.grid_mode == "legacy_attention":
            if "cross_attention" not in maps:
                raise ValueError(
                    "--grid_mode legacy_attention requires a cross_attention "
                    "checkpoint."
                )
            layer_maps = [
                maps[f"layer_{layer}_attention"] for layer in (3, 6, 9, 12)
            ]
            clip_final_map = maps["clip_final_cls_patch"]
            fusion_map = maps["cross_attention"]
        else:
            layer_maps = [
                maps[f"layer_{layer}_cls_relevance"]
                for layer in (3, 6, 9, 12)
            ]
            clip_final_map = maps["clip_final"]
            fusion_map = maps["loglo_fusion"]
        save_comparison_grid(
            original_image=image,
            layer3_visualization=layer_maps[0],
            layer6_visualization=layer_maps[1],
            layer9_visualization=layer_maps[2],
            layer12_visualization=layer_maps[3],
            clip_final_visualization=clip_final_map,
            fusion_visualization=fusion_map,
            rollout_visualization=maps["rollout"],
            caption=caption,
            output_path=args.grid_output,
            overlay_alpha=args.overlay_alpha,
            dpi=args.dpi,
            grid_mode=args.grid_mode,
        )
        print(f"Saved grid visualization to: {args.grid_output}")
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
