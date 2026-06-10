"""General utilities for reproducible LoGlo-CLIP experiments."""

import csv
import logging
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(name)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_data_parallel(model, device):
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def count_parameters(model):
    module = unwrap_model(model)
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return {"trainable": trainable, "total": total}


def trainable_parameters(model):
    return [
        parameter
        for parameter in unwrap_model(model).parameters()
        if parameter.requires_grad
    ]


def save_csv(rows, path, fieldnames=None):
    rows = list(rows)
    path = Path(path)
    ensure_dir(path.parent)
    if not rows and fieldnames is None:
        raise ValueError("fieldnames are required when saving an empty CSV.")
    fieldnames = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(row, path, identity_fields=()):
    path = Path(path)
    existing = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    if identity_fields:
        existing = [
            item
            for item in existing
            if any(str(item.get(key)) != str(row.get(key)) for key in identity_fields)
        ]
    existing.append(row)
    save_csv(existing, path)


def setup_logger(name, log_file=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def strip_module_prefix(state_dict):
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def safe_torch_load(path, map_location="cpu"):
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary.")
    for key in (
        "model_state_dict",
        "state_dict",
        "fusion_state_dict",
        "fusion_module_state_dict",
    ):
        if key in checkpoint:
            return strip_module_prefix(checkpoint[key])
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return strip_module_prefix(checkpoint)
    raise KeyError("Checkpoint does not contain a recognized model state dictionary.")


def format_model_name(model_type, layers=None):
    labels = {
        "clip": "CLIP ViT-B/32",
        "linear_l12": "CLIP + Linear(L12)",
        "weighted_sum_cls": "CLIP + Static WeightedSum-CLS",
        "weighted_sum_patch_mean": "CLIP + Static WeightedSum-PatchMean",
        "self_attention": "CLIP + Self-Attention Fusion",
        "cross_attention": "CLIP + Cross-Attention Fusion",
    }
    label = labels.get(model_type, model_type)
    if layers and model_type not in {"clip", "linear_l12"}:
        label += " [" + ",".join(map(str, layers)) + "]"
    return label


def slugify(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
