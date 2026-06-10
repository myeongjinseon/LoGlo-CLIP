"""Train and evaluate one Static WeightedSum-CLS layer combination."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loglo.data import create_flickr30k_loaders
from loglo.engine import evaluate_retrieval, load_checkpoint, save_checkpoint, train_one_epoch, validate
from loglo.metrics import flatten_retrieval_metrics
from loglo.models import build_model
from loglo.utils import append_csv, ensure_dir, format_model_name, get_device, set_seed, trainable_parameters


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="weighted_sum_cls")
    parser.add_argument("--dataset_name", default="nlphuji/flickr30k")
    parser.add_argument("--dataset_cache", type=Path)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_type != "weighted_sum_cls":
        raise ValueError("ablation_layers.py supports weighted_sum_cls only.")
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.save_dir)
    ensure_dir(args.result_dir)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    _, loaders = create_flickr30k_loaders(
        processor, args.dataset_name, args.dataset_cache,
        ("train", "val", "test"), args.batch_size, args.num_workers
    )
    model = build_model(args.model_type, args.clip_model, tuple(args.layers), args.alpha).to(device)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay
    )
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device, epoch)
        valid_loss = validate(model, loaders["val"], device)
        print(f"{args.layers} | {epoch}/{args.epochs} | train={train_loss:.4f} | val={valid_loss:.4f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_checkpoint(args.save_dir / "best.pt", model, optimizer, epoch, best_loss)
        save_checkpoint(args.save_dir / "last.pt", model, optimizer, epoch, best_loss)
    load_checkpoint(args.save_dir / "best.pt", model, map_location=device)
    metrics, _ = evaluate_retrieval(model, loaders["test"], device)
    tag = "_".join(map(str, args.layers))
    row = {
        "model": format_model_name(args.model_type, args.layers),
        "layers": " ".join(map(str, args.layers)),
        "checkpoint": str(args.save_dir / "best.pt"),
        **flatten_retrieval_metrics(metrics),
    }
    append_csv(
        row,
        args.result_dir / "ablation_layers.csv",
        identity_fields=("layers",),
    )
    weights = model.normalized_weights().detach().cpu().tolist()
    for layer, weight in zip(args.layers, weights):
        append_csv(
            {"layers": row["layers"], "layer": layer, "weight": weight},
            args.result_dir / "ablation_layer_weights.csv",
            identity_fields=("layers", "layer"),
        )
    print(row)


if __name__ == "__main__":
    main()
