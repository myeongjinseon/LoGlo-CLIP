"""Train the main CLIP + WeightedSum [3, 6, 9, 12] model."""

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
        count_parameters,
        create_split_loaders,
        ensure_output_dirs,
        load_checkpoint,
        load_clip,
        maybe_data_parallel,
        resolve_device,
        result_row,
        save_checkpoint,
        save_csv,
        train_one_epoch,
        unwrap_model,
        validate_layers,
        validation_loss,
    )
except ImportError:
    from common import (
        DEFAULT_ANNOTATIONS,
        MODEL_NAME,
        PROJECT_ROOT,
        count_parameters,
        create_split_loaders,
        ensure_output_dirs,
        load_checkpoint,
        load_clip,
        maybe_data_parallel,
        resolve_device,
        result_row,
        save_checkpoint,
        save_csv,
        train_one_epoch,
        unwrap_model,
        validate_layers,
        validation_loss,
    )


DEFAULT_LAYERS = (3, 6, 9, 12)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "weighted_sum_main"
RESULTS_PATH = PROJECT_ROOT / "results" / "weighted_sum_main_results.csv"
WEIGHTS_PATH = PROJECT_ROOT / "results" / "weighted_sum_main_layer_weights.csv"


class WeightedSumFusion(nn.Module):
    def __init__(self, num_layers=4, embed_dim=768, projection_dim=512):
        super().__init__()
        self.num_layers = num_layers
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.projector = nn.Linear(embed_dim, projection_dim)

    def forward(self, *hidden_states):
        if len(hidden_states) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} hidden states, got {len(hidden_states)}."
            )
        cls_tokens = torch.stack(
            [hidden_state[:, 0] for hidden_state in hidden_states], dim=1
        )
        weights = F.softmax(self.layer_logits, dim=0)
        fused = (cls_tokens * weights[None, :, None]).sum(dim=1)
        return F.normalize(self.projector(fused), dim=-1)

    def normalized_weights(self):
        return F.softmax(self.layer_logits, dim=0).detach().cpu().tolist()


def build_model(clip_model, layers=DEFAULT_LAYERS):
    return WeightedSumFusion(
        num_layers=len(layers),
        embed_dim=clip_model.config.vision_config.hidden_size,
        projection_dim=clip_model.config.projection_dim,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--residual_alpha", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    device = resolve_device(args.device)
    clip_model, processor = load_clip(args.model_name, device)
    layers = validate_layers(args.layers, clip_model)
    datasets, loaders = create_split_loaders(
        processor,
        args.annotations,
        args.batch_size,
        args.num_workers,
        ("train", "val"),
    )
    model = maybe_data_parallel(build_model(clip_model, layers).to(device), device)
    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    config = {
        "model_type": "weighted_sum",
        "model_name": args.model_name,
        "layers": list(layers),
        "residual_alpha": args.residual_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    best_path = CHECKPOINT_DIR / "best.pt"
    last_path = CHECKPOINT_DIR / "last.pt"
    best_loss = float("inf")
    trainable, total = count_parameters(clip_model, model)
    print(
        f"Device: {device} | Train: {len(datasets['train']):,} | "
        f"Validation: {len(datasets['val']):,}"
    )
    print(f"Layers: {list(layers)} | Residual alpha: {args.residual_alpha}")
    print(f"Trainable parameters: {trainable:,} | Total parameters: {total:,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            clip_model,
            processor,
            loaders["train"],
            optimizer,
            layers,
            device,
            args.residual_alpha,
        )
        valid_loss = validation_loss(
            model,
            clip_model,
            processor,
            loaders["val"],
            layers,
            device,
            args.residual_alpha,
        )
        weights = unwrap_model(model).normalized_weights()
        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Weights: {[round(weight, 4) for weight in weights]}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_checkpoint(
                best_path, epoch, model, optimizer, best_loss, config
            )
            print(f"Saved best checkpoint: {best_path}")
        save_checkpoint(last_path, epoch, model, optimizer, best_loss, config)

    checkpoint = load_checkpoint(best_path, model, map_location=device)
    weights = unwrap_model(model).normalized_weights()
    save_csv(
        [
            {
                "model": "CLIP+WeightedSum [3,6,9,12]",
                "trainable_params": trainable,
                "total_params": total,
                "best_valid_loss": checkpoint["best_valid_loss"],
                "best_epoch": checkpoint["epoch"],
                "checkpoint": best_path.relative_to(PROJECT_ROOT),
            }
        ],
        RESULTS_PATH,
    )
    save_csv(
        [
            {"layer": layer, "weight": f"{weight:.8f}"}
            for layer, weight in zip(layers, weights)
        ],
        WEIGHTS_PATH,
    )
    print(f"Best: {best_path} | Last: {last_path}")
    print(f"Results: {RESULTS_PATH} | Layer weights: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
