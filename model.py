from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleWiseFusionModule(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        projection_dim=512,
        pooling="cls",
        num_heads=8,
    ):
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'.")

        self.pooling = pooling
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.projector = nn.Linear(embed_dim, projection_dim)

    def forward(self, l3, l6, l9, l12, return_attention=False):
        local_features = torch.cat([l3, l6, l9], dim=1)
        attn_output, attention = self.cross_attn(
            query=l12,
            key=local_features,
            value=local_features,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        if self.pooling == "cls":
            pooled_feature = attn_output[:, 0, :]
        else:
            pooled_feature = attn_output.mean(dim=1)
        features = F.normalize(self.projector(pooled_feature), dim=-1)
        if return_attention:
            return features, attention
        return features


def unwrap_module(module):
    return module.module if isinstance(module, nn.DataParallel) else module


def strip_module_prefix(state_dict):
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def save_fusion_checkpoint(
    path,
    epoch,
    fusion_original_module,
    fusion_mean_module,
    optimizer,
    best_validation_loss,
    train_config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "fusion_original_module_state_dict": unwrap_module(
                fusion_original_module
            ).state_dict(),
            "fusion_mean_module_state_dict": unwrap_module(
                fusion_mean_module
            ).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "train_config": train_config,
        },
        path,
    )


def load_fusion_checkpoint(
    path,
    fusion_original_module,
    fusion_mean_module,
    optimizer=None,
    map_location="cpu",
):
    checkpoint = torch.load(path, map_location=map_location)
    original_key = (
        "fusion_original_module_state_dict"
        if "fusion_original_module_state_dict" in checkpoint
        else "fusion_original_module"
    )
    mean_key = (
        "fusion_mean_module_state_dict"
        if "fusion_mean_module_state_dict" in checkpoint
        else "fusion_mean_module"
    )
    unwrap_module(fusion_original_module).load_state_dict(
        strip_module_prefix(checkpoint[original_key])
    )
    unwrap_module(fusion_mean_module).load_state_dict(
        strip_module_prefix(checkpoint[mean_key])
    )

    if optimizer is not None:
        optimizer_state = checkpoint.get(
            "optimizer_state_dict",
            checkpoint.get("optimizer"),
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    return checkpoint
