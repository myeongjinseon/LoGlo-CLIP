"""Training, validation, embedding extraction, and checkpoint utilities."""

from pathlib import Path

import torch
from tqdm.auto import tqdm

from .losses import image_text_contrastive_loss
from .metrics import retrieval_metrics
from .utils import (
    ensure_dir,
    extract_state_dict,
    safe_torch_load,
    strip_module_prefix,
    unwrap_model,
)


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def train_one_epoch(model, loader, optimizer, device, epoch=None):
    model.train()
    total_loss = 0.0
    sample_count = 0
    progress = tqdm(loader, desc=f"Train {epoch}" if epoch else "Train")
    for batch in progress:
        batch = move_batch(batch, device)
        outputs = model(batch)
        module = unwrap_model(model)
        loss = image_text_contrastive_loss(
            outputs["image_embeddings"],
            outputs["text_embeddings"],
            logit_scale=module.clip.logit_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        size = batch["pixel_values"].size(0)
        total_loss += loss.item() * size
        sample_count += size
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(sample_count, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    sample_count = 0
    for batch in tqdm(loader, desc="Validate"):
        batch = move_batch(batch, device)
        outputs = model(batch)
        module = unwrap_model(model)
        loss = image_text_contrastive_loss(
            outputs["image_embeddings"],
            outputs["text_embeddings"],
            logit_scale=module.clip.logit_scale,
        )
        size = batch["pixel_values"].size(0)
        total_loss += loss.item() * size
        sample_count += size
    return total_loss / max(sample_count, 1)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    images, texts, ids, captions = [], [], [], []
    for batch in tqdm(loader, desc="Extract embeddings"):
        batch = move_batch(batch, device)
        outputs = model(batch)
        images.append(outputs["image_embeddings"].cpu())
        texts.append(outputs["text_embeddings"].cpu())
        ids.extend(batch["image_ids"])
        captions.extend(batch.get("captions", []))
    return {
        "image_embeddings": torch.cat(images),
        "text_embeddings": torch.cat(texts),
        "image_ids": ids,
        "captions": captions,
    }


def evaluate_retrieval(model, loader, device):
    embeddings = extract_embeddings(model, loader, device)
    return retrieval_metrics(
        embeddings["image_embeddings"], embeddings["text_embeddings"]
    ), embeddings


def save_checkpoint(
    path,
    model,
    optimizer=None,
    epoch=None,
    best_valid_loss=None,
    extra=None,
):
    path = Path(path)
    ensure_dir(path.parent)
    module = unwrap_model(model)
    state_dict = {
        key: value
        for key, value in module.state_dict().items()
        if not key.startswith("clip.")
    }
    payload = {
        "model_state_dict": state_dict,
        "config": module.checkpoint_config(),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu", strict=True):
    checkpoint = safe_torch_load(path, map_location)
    state_dict = extract_state_dict(checkpoint)
    module = unwrap_model(model)
    try:
        module.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        # Older checkpoints stored only the trainable fusion module.
        trainable_keys = {
            key for key, parameter in module.named_parameters() if parameter.requires_grad
        }
        filtered = {
            key: value
            for key, value in strip_module_prefix(state_dict).items()
            if key in trainable_keys or not key.startswith("clip.")
        }
        module.load_state_dict(filtered, strict=False)
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
