"""Loss functions used by LoGlo-CLIP."""

import torch
import torch.nn.functional as F


def image_text_contrastive_loss(
    image_embeddings,
    text_embeddings,
    temperature=None,
    logit_scale=None,
):
    if logit_scale is None:
        temperature = 0.07 if temperature is None else temperature
        scale = 1.0 / temperature
    else:
        scale = logit_scale.exp() if torch.is_tensor(logit_scale) else logit_scale
    logits = scale * image_embeddings @ text_embeddings.T
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )
