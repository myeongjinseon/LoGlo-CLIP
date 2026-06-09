"""Evaluate a fusion checkpoint against the original CLIP ViT-B/32."""

import argparse
from pathlib import Path

import torch

try:
    from .common import (
        DEFAULT_ANNOTATIONS,
        PROJECT_ROOT,
        checkpoint_config,
        count_parameters,
        create_split_loaders,
        ensure_output_dirs,
        evaluate_features,
        load_checkpoint,
        load_clip,
        print_results,
        resolve_device,
        result_row,
        save_csv,
        validate_layers,
    )
    from .train import build_model as build_weighted_sum
except ImportError:
    from common import (
        DEFAULT_ANNOTATIONS,
        PROJECT_ROOT,
        checkpoint_config,
        count_parameters,
        create_split_loaders,
        ensure_output_dirs,
        evaluate_features,
        load_checkpoint,
        load_clip,
        print_results,
        resolve_device,
        result_row,
        save_csv,
        validate_layers,
    )
    from train import build_model as build_weighted_sum


RESULTS_PATH = PROJECT_ROOT / "results" / "evaluation_results.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/weighted_sum_main/best.pt"),
    )
    parser.add_argument("--model_type", choices=("weighted_sum",), default="weighted_sum")
    parser.add_argument("--layers", type=int, nargs="+", default=(3, 6, 9, 12))
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    checkpoint_path = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else PROJECT_ROOT / args.checkpoint
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    metadata = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint_config(metadata)
    device = resolve_device(args.device)
    clip_model, processor = load_clip(
        config.get("model_name", "openai/clip-vit-base-patch32"), device
    )
    layers = validate_layers(config.get("layers", args.layers), clip_model)
    alpha = float(config.get("residual_alpha", 0.3))
    model = build_weighted_sum(clip_model, layers).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    datasets, loaders = create_split_loaders(
        processor,
        args.annotations,
        args.batch_size,
        args.num_workers,
        ("test",),
    )
    recalls = evaluate_features(
        clip_model,
        processor,
        loaders["test"],
        device,
        layers,
        model,
        alpha,
    )
    rows = [
        result_row(
            "CLIP ViT-B/32",
            (0, count_parameters(clip_model)[1]),
            recalls["clip"],
        ),
        result_row(
            f"CLIP+WeightedSum {list(layers)}",
            count_parameters(clip_model, model),
            recalls["residual"],
            metadata.get("best_valid_loss", ""),
            checkpoint_path.relative_to(PROJECT_ROOT),
        ),
    ]
    print(f"Test samples: {len(datasets['test']):,} | Device: {device}")
    print_results(rows)
    save_csv(rows, RESULTS_PATH)
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
