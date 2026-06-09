"""WeightedSum layer ablation with CLIP and Linear(L12) controls."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .common import (
        DEFAULT_ANNOTATIONS,
        MODEL_NAME,
        PROJECT_ROOT,
        clip_validation_loss,
        count_parameters,
        create_split_loaders,
        ensure_output_dirs,
        evaluate_features,
        load_checkpoint,
        load_clip,
        maybe_data_parallel,
        print_results,
        resolve_device,
        result_row,
        save_checkpoint,
        save_csv,
        train_one_epoch,
        unwrap_model,
        validate_layers,
        validation_loss,
    )
    from .train import WeightedSumFusion
except ImportError:
    from common import (
        DEFAULT_ANNOTATIONS,
        MODEL_NAME,
        PROJECT_ROOT,
        clip_validation_loss,
        count_parameters,
        create_split_loaders,
        ensure_output_dirs,
        evaluate_features,
        load_checkpoint,
        load_clip,
        maybe_data_parallel,
        print_results,
        resolve_device,
        result_row,
        save_checkpoint,
        save_csv,
        train_one_epoch,
        unwrap_model,
        validate_layers,
        validation_loss,
    )
    from train import WeightedSumFusion


DEFAULT_COMBINATIONS = ((12,), (9, 12), (6, 9, 12), (3, 6, 9, 12))
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "layer_ablation"
RESULTS_PATH = PROJECT_ROOT / "results" / "layer_ablation_results.csv"
WEIGHTS_PATH = PROJECT_ROOT / "results" / "layer_ablation_weights.csv"


class LinearL12(nn.Module):
    def __init__(self, embed_dim, projection_dim):
        super().__init__()
        self.projector = nn.Linear(embed_dim, projection_dim)

    def forward(self, hidden_state):
        return F.normalize(self.projector(hidden_state[:, 0]), dim=-1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--residual_alpha", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--include_optional", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def train_variant(name, key, model, layers, clip_model, processor, loaders, device, args):
    model = maybe_data_parallel(model.to(device), device)
    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_loss = float("inf")
    best_path = CHECKPOINT_DIR / key / "best.pt"
    last_path = CHECKPOINT_DIR / key / "last.pt"
    config = {
        "model_type": "weighted_sum" if isinstance(unwrap_model(model), WeightedSumFusion) else "linear_l12",
        "experiment_model": name,
        "model_name": args.model_name,
        "layers": list(layers),
        "residual_alpha": args.residual_alpha,
    }
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, clip_model, processor, loaders["train"], optimizer,
            layers, device, args.residual_alpha,
        )
        valid_loss = validation_loss(
            model, clip_model, processor, loaders["val"],
            layers, device, args.residual_alpha,
        )
        print(
            f"{name} | Epoch {epoch}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Valid: {valid_loss:.4f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_checkpoint(best_path, epoch, model, optimizer, best_loss, config)
        save_checkpoint(last_path, epoch, model, optimizer, best_loss, config)
    load_checkpoint(best_path, model, map_location=device)
    recalls = evaluate_features(
        clip_model, processor, loaders["test"], device,
        layers, model, args.residual_alpha,
    )["residual"]
    row = result_row(
        name,
        count_parameters(clip_model, model),
        recalls,
        best_loss,
        best_path.relative_to(PROJECT_ROOT),
    )
    return model, row


def main():
    args = parse_args()
    ensure_output_dirs()
    device = resolve_device(args.device)
    clip_model, processor = load_clip(args.model_name, device)
    combinations = list(DEFAULT_COMBINATIONS)
    if args.include_optional:
        combinations.append((2, 5, 8, 11))
    combinations = [validate_layers(layers, clip_model) for layers in combinations]
    _, loaders = create_split_loaders(
        processor,
        args.annotations,
        args.batch_size,
        args.num_workers,
        ("train", "val", "test"),
    )
    clip_recalls = evaluate_features(
        clip_model, processor, loaders["test"], device
    )["clip"]
    rows = [
        result_row(
            "CLIP ViT-B/32",
            (0, count_parameters(clip_model)[1]),
            clip_recalls,
            clip_validation_loss(clip_model, processor, loaders["val"], device),
        )
    ]
    weights = []
    embed_dim = clip_model.config.vision_config.hidden_size
    projection_dim = clip_model.config.projection_dim

    _, row = train_variant(
        "CLIP+Linear(L12)",
        "linear_l12",
        LinearL12(embed_dim, projection_dim),
        (12,),
        clip_model,
        processor,
        loaders,
        device,
        args,
    )
    rows.append(row)
    save_csv(rows, RESULTS_PATH)

    for layers in combinations:
        label = "[" + ",".join(map(str, layers)) + "]"
        key = "weighted_sum_" + "_".join(map(str, layers))
        model, row = train_variant(
            f"CLIP+WeightedSum {label}",
            key,
            WeightedSumFusion(len(layers), embed_dim, projection_dim),
            layers,
            clip_model,
            processor,
            loaders,
            device,
            args,
        )
        rows.append(row)
        weights.extend(
            {
                "model": row["model"],
                "layers": label,
                "layer": layer,
                "weight": f"{weight:.8f}",
            }
            for layer, weight in zip(
                layers, unwrap_model(model).normalized_weights()
            )
        )
        save_csv(rows, RESULTS_PATH)
        save_csv(
            weights,
            WEIGHTS_PATH,
            ("model", "layers", "layer", "weight"),
        )
        print_results(rows)
    print(f"Saved: {RESULTS_PATH}")
    print(f"Saved: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
