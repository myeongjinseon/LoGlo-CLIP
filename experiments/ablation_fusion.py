"""Train and evaluate one fusion method for the fusion ablation."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loglo.data import create_flickr30k_loaders
from loglo.engine import evaluate_retrieval, load_checkpoint, save_checkpoint, train_one_epoch, validate
from loglo.metrics import flatten_retrieval_metrics
from loglo.models import MODEL_REGISTRY, build_model
from loglo.utils import append_csv, ensure_dir, format_model_name, get_device, set_seed, trainable_parameters


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", choices=sorted(MODEL_REGISTRY), required=True)
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
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--experiment_name", default="ablation_fusion")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main():
    args = parse_args()
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
    best_path = args.save_dir / "best.pt"
    if args.model_type != "clip":
        optimizer = torch.optim.AdamW(
            trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay
        )
        best_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, loaders["train"], optimizer, device, epoch)
            valid_loss = validate(model, loaders["val"], device)
            print(f"{args.model_type} | {epoch}/{args.epochs} | train={train_loss:.4f} | val={valid_loss:.4f}")
            if valid_loss < best_loss:
                best_loss = valid_loss
                save_checkpoint(best_path, model, optimizer, epoch, best_loss)
            save_checkpoint(args.save_dir / "last.pt", model, optimizer, epoch, best_loss)
        load_checkpoint(best_path, model, map_location=device)
    metrics, _ = evaluate_retrieval(model, loaders["test"], device)
    row = {
        "model_type": args.model_type,
        "model": format_model_name(args.model_type, args.layers),
        "checkpoint": str(best_path) if args.model_type != "clip" else "",
        **flatten_retrieval_metrics(metrics),
    }
    output = args.result_dir / f"{args.experiment_name}.csv"
    append_csv(row, output, identity_fields=("model_type",))
    print(row)


if __name__ == "__main__":
    main()
