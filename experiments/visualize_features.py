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
    save_fusion_weights,
    save_heatmap_overlay,
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
    parser.add_argument("--output_dir", type=Path, default=Path("visualization/weighted_sum_cls"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def select_image(args):
    if args.image_path:
        if not args.image_path.is_file():
            raise FileNotFoundError(f"Image not found: {args.image_path}")
        with Image.open(args.image_path) as image:
            return image.convert("RGB")
    raw = select_split(load_flickr30k(args.dataset_name, args.dataset_cache), args.split)
    dataset = Flickr30kDataset(raw, args.split)
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"--index must be between 0 and {len(dataset) - 1}.")
    return dataset[args.index]["image"]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.output_dir)
    image = select_image(args)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = build_model(
        args.model_type, args.clip_model, tuple(args.layers), args.alpha
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    pixels = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    maps = extract_visual_maps(model, pixels, args.layers)
    save_original_image(image, args.output_dir / "original.png")
    for name, heatmap in maps.items():
        save_heatmap_overlay(
            image,
            heatmap,
            args.output_dir / f"{name}.png",
            name.replace("_", " ").title(),
            dpi=args.dpi,
        )
    save_fusion_weights(model, args.output_dir / "fusion_weights.png", args.dpi)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
