"""Evaluate the frozen original CLIP ViT-B/32 on Flickr30k test."""

import argparse
from pathlib import Path

try:
    from .common import (
        DEFAULT_ANNOTATIONS,
        MODEL_NAME,
        PROJECT_ROOT,
        count_parameters,
        create_split_loaders,
        ensure_output_dirs,
        evaluate_features,
        load_clip,
        print_results,
        resolve_device,
        result_row,
        save_csv,
    )
except ImportError:
    from common import (
        DEFAULT_ANNOTATIONS,
        MODEL_NAME,
        PROJECT_ROOT,
        count_parameters,
        create_split_loaders,
        ensure_output_dirs,
        evaluate_features,
        load_clip,
        print_results,
        resolve_device,
        result_row,
        save_csv,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    device = resolve_device(args.device)
    clip_model, processor = load_clip(args.model_name, device)
    datasets, loaders = create_split_loaders(
        processor, args.annotations, args.batch_size, args.num_workers, ("test",)
    )
    recalls = evaluate_features(
        clip_model, processor, loaders["test"], device
    )["clip"]
    row = result_row(
        "CLIP ViT-B/32", (0, count_parameters(clip_model)[1]), recalls
    )
    print(f"Test samples: {len(datasets['test']):,} | Device: {device}")
    print_results([row])
    path = PROJECT_ROOT / "results" / "original_clip_results.csv"
    save_csv([row], path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
