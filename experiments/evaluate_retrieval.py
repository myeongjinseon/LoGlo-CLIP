"""Evaluate a CLIP or LoGlo checkpoint on image-text retrieval."""

import argparse
import sys
from pathlib import Path

from transformers import CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loglo.data import create_flickr30k_loaders
from loglo.engine import evaluate_retrieval, load_checkpoint
from loglo.metrics import flatten_retrieval_metrics
from loglo.models import MODEL_REGISTRY, build_model
from loglo.utils import (
    count_parameters,
    ensure_dir,
    format_model_name,
    get_device,
    save_csv,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", choices=sorted(MODEL_REGISTRY), default="weighted_sum_cls")
    parser.add_argument("--dataset_name", default="nlphuji/flickr30k")
    parser.add_argument("--dataset_cache", type=Path)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--layers", type=int, nargs="+", default=(3, 6, 9, 12))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--result_dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.result_dir)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    datasets, loaders = create_flickr30k_loaders(
        processor,
        args.dataset_name,
        args.dataset_cache,
        (args.split,),
        args.batch_size,
        args.num_workers,
    )
    model = build_model(
        args.model_type, args.clip_model, tuple(args.layers), args.alpha
    ).to(device)
    if args.model_type != "clip":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for trainable model types.")
        load_checkpoint(args.checkpoint, model, map_location=device)
    metrics, _ = evaluate_retrieval(model, loaders[args.split], device)
    params = count_parameters(model)
    row = {
        "model": format_model_name(args.model_type, args.layers),
        "split": args.split,
        "samples": len(datasets[args.split]),
        "checkpoint": str(args.checkpoint or ""),
        "trainable_params": params["trainable"],
        "total_params": params["total"],
        **flatten_retrieval_metrics(metrics),
    }
    save_csv([row], args.result_dir / "evaluation_results.csv")
    print(row)


if __name__ == "__main__":
    main()
