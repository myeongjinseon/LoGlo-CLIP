"""Shared Flickr30k, CLIP, training, evaluation, and persistence utilities."""

import csv
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "openai/clip-vit-base-patch32"
DATASET_NAME = "nlphuji/flickr30k"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "flickr30k" / "annotations.csv"
OUTPUT_DIRS = ("checkpoints", "results", "logs", "visualization")
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "validation"),
    "validation": ("validation", "val"),
    "test": ("test",),
}


def ensure_output_dirs():
    for name in OUTPUT_DIRS:
        (PROJECT_ROOT / name).mkdir(parents=True, exist_ok=True)


def resolve_project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_split(split):
    return "val" if split == "validation" else split


def get_huggingface_split(dataset_dict, split):
    aliases = SPLIT_ALIASES.get(split, (split,))
    physical_splits = list(dataset_dict)
    if len(physical_splits) == 1:
        dataset = dataset_dict[physical_splits[0]]
    else:
        physical = next((name for name in aliases if name in dataset_dict), None)
        if physical is None:
            raise ValueError(
                f"Split {split!r} is unavailable. Found: {physical_splits}"
            )
        dataset = dataset_dict[physical]

    if "split" not in dataset.column_names:
        return dataset
    values = dataset["split"]
    internal = next((name for name in aliases if name in set(values)), None)
    if internal is None:
        if len(physical_splits) == 1:
            raise ValueError(
                f"Split {split!r} is unavailable. Found: {sorted(set(values))}"
            )
        return dataset
    return dataset.select([index for index, value in enumerate(values) if value == internal])


def _select_caption(value):
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value or "")


def _safe_stem(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(value)).stem).strip("._")


def _prepare_local_cache(annotations_path, dataset_name=DATASET_NAME):
    from datasets import load_dataset

    annotations_path = Path(annotations_path)
    image_root = annotations_path.parent / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    dataset_dict = load_dataset(dataset_name)
    rows = []
    for split in ("train", "val", "test"):
        dataset = get_huggingface_split(dataset_dict, split)
        split_dir = image_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc=f"Caching Flickr30k {split}",
        ):
            source_id = next(
                (
                    sample.get(key)
                    for key in ("filename", "file_name", "img_id", "image_id")
                    if sample.get(key) is not None
                ),
                index,
            )
            filename = f"{index:06d}_{_safe_stem(source_id)}.jpg"
            image_path = split_dir / filename
            if not image_path.exists():
                sample["image"].convert("RGB").save(image_path, quality=95)
            try:
                stored_path = image_path.relative_to(PROJECT_ROOT)
            except ValueError:
                stored_path = image_path
            rows.append(
                {
                    "split_index": index,
                    "image_filename": filename,
                    "image_path": stored_path.as_posix(),
                    "caption": _select_caption(sample.get("caption")),
                    "split": split,
                }
            )
    save_csv(rows, annotations_path)


def load_records(annotations_path=DEFAULT_ANNOTATIONS, split=None):
    annotations_path = resolve_project_path(annotations_path)
    if not annotations_path.exists():
        _prepare_local_cache(annotations_path)
    with annotations_path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if split is None:
        return records
    split = normalize_split(split)
    return [
        row for row in records if normalize_split(row.get("split", "")) == split
    ]


def resolve_image_path(path, annotations_path=DEFAULT_ANNOTATIONS):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    annotations_path = resolve_project_path(annotations_path)
    candidates = (PROJECT_ROOT / path, annotations_path.parent / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


class Flickr30kDataset(Dataset):
    """Flickr30k image-caption pairs backed by the HuggingFace dataset cache."""

    def __init__(self, processor, split="train", annotations_path=DEFAULT_ANNOTATIONS):
        self.processor = processor
        self.split = normalize_split(split)
        self.annotations_path = resolve_project_path(annotations_path)
        self.records = load_records(self.annotations_path, self.split)
        if not self.records:
            raise ValueError(f"No Flickr30k samples found for split {self.split!r}.")

    def __len__(self):
        return len(self.records)

    def get_record(self, index):
        record = dict(self.records[index])
        record["image_path"] = resolve_image_path(
            record["image_path"], self.annotations_path
        )
        return record

    def __getitem__(self, index):
        record = self.get_record(index)
        with Image.open(record["image_path"]) as image:
            pixel_values = self.processor(
                images=image.convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]
        return pixel_values, record["caption"]


def create_dataloader(
    dataset, batch_size=64, shuffle=False, num_workers=None, pin_memory=None
):
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def create_split_loaders(processor, annotations, batch_size, num_workers, splits):
    datasets = {
        split: Flickr30kDataset(processor, split, annotations) for split in splits
    }
    loaders = {
        split: create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders


def resolve_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def load_clip(model_name=MODEL_NAME, device="cpu", eager_attention=False):
    kwargs = {"attn_implementation": "eager"} if eager_attention else {}
    try:
        processor = CLIPProcessor.from_pretrained(
            model_name, local_files_only=True
        )
    except OSError:
        processor = CLIPProcessor.from_pretrained(model_name)
    try:
        try:
            model = CLIPModel.from_pretrained(
                model_name, local_files_only=True, **kwargs
            )
        except TypeError:
            model = CLIPModel.from_pretrained(
                model_name, local_files_only=True
            )
    except OSError:
        try:
            model = CLIPModel.from_pretrained(model_name, **kwargs)
        except TypeError:
            model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.requires_grad_(False)
    model.eval()
    return model, processor


def validate_layers(layers, clip_model):
    layers = tuple(int(layer) for layer in layers)
    if not layers or tuple(sorted(set(layers))) != layers:
        raise ValueError("Layers must be unique and sorted in ascending order.")
    maximum = clip_model.config.vision_config.num_hidden_layers
    if layers[0] < 1 or layers[-1] > maximum:
        raise ValueError(f"Vision layers must be between 1 and {maximum}.")
    return layers


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_data_parallel(model, device):
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
        return nn.DataParallel(model)
    return model


def count_parameters(*models):
    total = 0
    trainable = 0
    for model in models:
        if model is None:
            continue
        module = unwrap_model(model)
        total += sum(parameter.numel() for parameter in module.parameters())
        trainable += sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    return trainable, total


def contrastive_loss(image_features, text_features, logit_scale):
    logits = image_features @ text_features.T * logit_scale
    labels = torch.arange(image_features.size(0), device=image_features.device)
    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    ) / 2


def compute_recall(image_features, text_features):
    similarity = image_features @ text_features.T
    count = similarity.size(0)
    if count == 0:
        raise ValueError("Cannot compute recall for an empty dataset.")
    targets = torch.arange(count, device=similarity.device)
    maximum = min(10, count)
    i2t = similarity.topk(maximum, dim=1).indices
    t2i = similarity.T.topk(maximum, dim=1).indices

    def recall(retrieved, k):
        matches = retrieved[:, : min(k, count)].eq(targets[:, None]).any(dim=1)
        return matches.float().mean().item() * 100

    return {
        "i2t": {k: recall(i2t, k) for k in (1, 5, 10)},
        "t2i": {k: recall(t2i, k) for k in (1, 5, 10)},
    }


@torch.no_grad()
def encode_frozen_clip(
    clip_model, processor, images, captions, layers, device
):
    clip_model.eval()
    vision = clip_model.vision_model(
        pixel_values=images, output_hidden_states=True, return_dict=True
    )
    hidden_states = [vision.hidden_states[layer] for layer in layers]
    clip_features = F.normalize(
        clip_model.visual_projection(vision.pooler_output), dim=-1
    )
    text_inputs = processor(
        text=list(captions),
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    text = clip_model.text_model(**text_inputs, return_dict=True)
    text_features = F.normalize(
        clip_model.text_projection(text.pooler_output), dim=-1
    )
    return hidden_states, clip_features, text_features


def residual_feature(clip_feature, fusion_feature, alpha):
    return F.normalize(clip_feature + alpha * fusion_feature, dim=-1)


def save_checkpoint(
    path, epoch, model, optimizer, best_valid_loss, config, **extra
):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "best_valid_loss": best_valid_loss,
        "config": config,
    }
    payload.update(extra)
    torch.save(payload, path)


def _checkpoint_state(checkpoint):
    for key in (
        "model_state_dict",
        "head_state_dict",
        "fusion_model_state_dict",
        "weighted_sum_state_dict",
    ):
        if key in checkpoint:
            return checkpoint[key]
    raise KeyError("Checkpoint does not contain a supported model state dict.")


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(resolve_project_path(path), map_location=map_location)
    state = {
        key.removeprefix("module."): value
        for key, value in _checkpoint_state(checkpoint).items()
    }
    unwrap_model(model).load_state_dict(state)
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def checkpoint_config(checkpoint):
    return checkpoint.get("config", checkpoint.get("train_config", {}))


def train_one_epoch(
    model, clip_model, processor, dataloader, optimizer, layers, device, alpha
):
    model.train()
    loss_sum = 0.0
    sample_count = 0
    progress = tqdm(dataloader, desc="Training", unit="batch")
    for images, captions in progress:
        images = images.to(device, non_blocking=True)
        hidden, clip_features, text_features = encode_frozen_clip(
            clip_model, processor, images, captions, layers, device
        )
        image_features = residual_feature(
            clip_features, model(*hidden), alpha
        )
        loss = contrastive_loss(
            image_features, text_features, clip_model.logit_scale.exp()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}")
    if sample_count == 0:
        raise ValueError("Training dataloader is empty.")
    return loss_sum / sample_count


@torch.no_grad()
def validation_loss(
    model, clip_model, processor, dataloader, layers, device, alpha
):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    for images, captions in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        hidden, clip_features, text_features = encode_frozen_clip(
            clip_model, processor, images, captions, layers, device
        )
        image_features = residual_feature(clip_features, model(*hidden), alpha)
        loss = contrastive_loss(
            image_features, text_features, clip_model.logit_scale.exp()
        )
        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Validation dataloader is empty.")
    return loss_sum / sample_count


@torch.no_grad()
def clip_validation_loss(clip_model, processor, dataloader, device):
    clip_model.eval()
    loss_sum = 0.0
    sample_count = 0
    for images, captions in tqdm(
        dataloader, desc="CLIP validation", leave=False
    ):
        images = images.to(device, non_blocking=True)
        _, clip_features, text_features = encode_frozen_clip(
            clip_model, processor, images, captions, (12,), device
        )
        loss = contrastive_loss(
            clip_features, text_features, clip_model.logit_scale.exp()
        )
        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Validation dataloader is empty.")
    return loss_sum / sample_count


@torch.no_grad()
def evaluate_features(
    clip_model,
    processor,
    dataloader,
    device,
    layers=(12,),
    fusion_model=None,
    alpha=0.3,
    include_fusion_only=False,
):
    if fusion_model is not None:
        fusion_model.eval()
    clip_batches = []
    fusion_batches = []
    residual_batches = []
    text_batches = []
    for images, captions in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        hidden, clip_features, text_features = encode_frozen_clip(
            clip_model, processor, images, captions, layers, device
        )
        clip_batches.append(clip_features.cpu())
        text_batches.append(text_features.cpu())
        if fusion_model is not None:
            fusion = fusion_model(*hidden)
            if include_fusion_only:
                fusion_batches.append(fusion.cpu())
            residual_batches.append(residual_feature(clip_features, fusion, alpha).cpu())
    text_features = torch.cat(text_batches)
    results = {
        "clip": compute_recall(torch.cat(clip_batches), text_features)
    }
    if residual_batches:
        results["residual"] = compute_recall(
            torch.cat(residual_batches), text_features
        )
    if fusion_batches:
        results["fusion"] = compute_recall(torch.cat(fusion_batches), text_features)
    return results


def result_row(model, parameters, recalls, valid_loss="", checkpoint=""):
    trainable, total = parameters
    return {
        "model": model,
        "trainable_params": trainable,
        "total_params": total,
        "valid_loss": valid_loss,
        "i2t_r1": recalls["i2t"][1],
        "i2t_r5": recalls["i2t"][5],
        "i2t_r10": recalls["i2t"][10],
        "t2i_r1": recalls["t2i"][1],
        "t2i_r5": recalls["t2i"][5],
        "t2i_r10": recalls["t2i"][10],
        "checkpoint": str(checkpoint),
    }


def save_csv(rows, path, fieldnames=None):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames are required when saving no rows.")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_results(rows):
    print(
        f"{'Model':<30} | {'Trainable Params':>16} | {'Total Params':>13} | "
        f"{'I2T R@1':>7} {'R@5':>7} {'R@10':>7} | "
        f"{'T2I R@1':>7} {'R@5':>7} {'R@10':>7}"
    )
    print("-" * 122)
    for row in rows:
        print(
            f"{row['model']:<30} | {int(row['trainable_params']):16,} | "
            f"{int(row['total_params']):13,} | "
            f"{float(row['i2t_r1']):7.2f} {float(row['i2t_r5']):7.2f} "
            f"{float(row['i2t_r10']):7.2f} | "
            f"{float(row['t2i_r1']):7.2f} {float(row['t2i_r5']):7.2f} "
            f"{float(row['t2i_r10']):7.2f}"
        )


def run_fusion_experiment(
    *,
    model,
    display_names,
    checkpoint_dir,
    results_path,
    args,
    layers,
    include_fusion_only=True,
):
    """Train one fusion head and save CLIP/fusion/residual test results."""
    ensure_output_dirs()
    device = resolve_device(args.device)
    clip_model, processor = load_clip(args.model_name, device)
    layers = validate_layers(layers, clip_model)
    datasets, loaders = create_split_loaders(
        processor,
        args.annotations,
        args.batch_size,
        args.num_workers,
        ("train", "val", "test"),
    )
    model = maybe_data_parallel(model(clip_model).to(device), device)
    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    config = {
        "model_type": display_names["type"],
        "model_name": args.model_name,
        "layers": list(layers),
        "residual_alpha": args.residual_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    checkpoint_dir = resolve_project_path(checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    best_loss = float("inf")
    print(
        f"Device: {device} | Train: {len(datasets['train']):,} | "
        f"Validation: {len(datasets['val']):,} | Test: {len(datasets['test']):,}"
    )
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
        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_checkpoint(best_path, epoch, model, optimizer, best_loss, config)
            print(f"Saved best checkpoint: {best_path}")
        save_checkpoint(last_path, epoch, model, optimizer, best_loss, config)

    load_checkpoint(best_path, model, map_location=device)
    recalls = evaluate_features(
        clip_model,
        processor,
        loaders["test"],
        device,
        layers,
        model,
        args.residual_alpha,
        include_fusion_only,
    )
    clip_total = count_parameters(clip_model)[1]
    combined = count_parameters(clip_model, model)
    rows = [
        result_row("CLIP ViT-B/32", (0, clip_total), recalls["clip"])
    ]
    if include_fusion_only:
        rows.append(
            result_row(
                display_names["fusion"],
                count_parameters(model),
                recalls["fusion"],
                best_loss,
                best_path.relative_to(PROJECT_ROOT),
            )
        )
    rows.append(
        result_row(
            display_names["residual"],
            combined,
            recalls["residual"],
            best_loss,
            best_path.relative_to(PROJECT_ROOT),
        )
    )
    print_results(rows)
    save_csv(rows, results_path)
    print(f"Best: {best_path} | Last: {last_path}")
    print(f"Saved: {resolve_project_path(results_path)}")
    return rows
