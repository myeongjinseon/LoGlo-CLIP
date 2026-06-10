"""Train the main Static WeightedSum-CLS model and evaluate retrieval."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loglo.data import create_flickr30k_loaders
from loglo.engine import (
    evaluate_retrieval,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate,
)
from loglo.metrics import flatten_retrieval_metrics
from loglo.models import build_model
from loglo.utils import (
    count_parameters,
    ensure_dir,
    format_model_name,
    get_device,
    maybe_data_parallel,
    save_csv,
    set_seed,
    trainable_parameters,
    unwrap_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="weighted_sum_cls")
    parser.add_argument("--dataset_name", default="nlphuji/flickr30k")
    parser.add_argument("--dataset_cache", type=Path)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--layers", type=int, nargs="+", default=(3, 6, 9, 12))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints/weighted_sum_cls"))
    parser.add_argument("--result_dir", type=Path, default=Path("results/weighted_sum_cls"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_type != "weighted_sum_cls":
        raise ValueError("train_weighted_sum.py only supports weighted_sum_cls.")
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.save_dir)
    ensure_dir(args.result_dir)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    datasets, loaders = create_flickr30k_loaders(
        processor,
        args.dataset_name,
        args.dataset_cache,
        ("train", "val", "test"),
        args.batch_size,
        args.num_workers,
    )
    model = build_model(
        args.model_type, args.clip_model, tuple(args.layers), args.alpha
    ).to(device)
    model = maybe_data_parallel(model, device)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay
    )
    best_path = args.save_dir / "best.pt"
    last_path = args.save_dir / "last.pt"
    best_loss = float("inf")
    print(
        f"Device: {device} | Train/Val/Test: "
        f"{len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device, epoch)
        valid_loss = validate(model, loaders["val"], device)
        print(
            f"Epoch {epoch}/{args.epochs} | train={train_loss:.4f} "
            f"| val={valid_loss:.4f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_checkpoint(best_path, model, optimizer, epoch, best_loss)
        save_checkpoint(last_path, model, optimizer, epoch, best_loss)

    metadata = load_checkpoint(best_path, model, map_location=device)
    metrics, _ = evaluate_retrieval(model, loaders["test"], device)
    params = count_parameters(model)
    row = {
        "model": format_model_name(args.model_type, args.layers),
        "checkpoint": str(best_path),
        "best_epoch": metadata.get("epoch"),
        "best_valid_loss": metadata.get("best_valid_loss"),
        "trainable_params": params["trainable"],
        "total_params": params["total"],
        **flatten_retrieval_metrics(metrics),
    }
    save_csv([row], args.result_dir / "main_retrieval_results.csv")
    weights = unwrap_model(model).normalized_weights().detach().cpu().tolist()
    save_csv(
        [
            {"layer": layer, "weight": weight}
            for layer, weight in zip(args.layers, weights)
        ],
        args.result_dir / "learned_layer_weights.csv",
    )
    print(row)


if __name__ == "__main__":
    main()
