"""Train CLIP+LoGlo-V2 Self-Attention Fusion on layers [3, 6, 9, 12]."""

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
        run_fusion_experiment,
    )
except ImportError:
    from common import (
        DEFAULT_ANNOTATIONS,
        MODEL_NAME,
        PROJECT_ROOT,
        run_fusion_experiment,
    )


LAYERS = (3, 6, 9, 12)


class SelfAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, projection_dim=512, num_heads=8):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.fuse_cls = nn.Linear(embed_dim * len(LAYERS), embed_dim)
        self.projector = nn.Linear(embed_dim, projection_dim)

    def forward(self, *hidden_states):
        sequence_length = hidden_states[0].size(1)
        tokens = torch.cat(hidden_states, dim=1)
        attended, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        cls_tokens = [
            attended[:, index * sequence_length]
            for index in range(len(hidden_states))
        ]
        fused = self.fuse_cls(torch.cat(cls_tokens, dim=-1))
        return F.normalize(self.projector(fused), dim=-1)


def build_model(clip_model):
    return SelfAttentionFusion(
        clip_model.config.vision_config.hidden_size,
        clip_model.config.projection_dim,
    )


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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    run_fusion_experiment(
        model=build_model,
        display_names={
            "type": "self_attention",
            "fusion": "SelfAttention-only",
            "residual": "CLIP+SelfAttention",
        },
        checkpoint_dir=PROJECT_ROOT / "checkpoints" / "self_attention",
        results_path=PROJECT_ROOT / "results" / "self_attention_results.csv",
        args=parse_args(),
        layers=LAYERS,
    )


if __name__ == "__main__":
    main()
