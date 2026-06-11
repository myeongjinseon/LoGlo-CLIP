"""Flickr30k loading, preprocessing, tokenization, and data loaders."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET = "nlphuji/flickr30k"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_ANNOTATIONS = PROJECT_ROOT / "data" / "flickr30k" / "annotations.csv"
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "validation"),
    "validation": ("validation", "val"),
    "test": ("test",),
}


def normalize_split(split):
    return "val" if split == "validation" else split


class LocalFlickrSplit(Dataset):
    """Minimal HuggingFace-like split backed by prepared local images."""

    column_names = ("image", "caption", "image_id", "split")

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, str):
            if index not in self.column_names:
                raise KeyError(index)
            if index == "split":
                return [normalize_split(row["split"]) for row in self.rows]
            if index == "caption":
                return [row["caption"] for row in self.rows]
            if index == "image_id":
                return [
                    row.get("image_filename", Path(row["image_path"]).name)
                    for row in self.rows
                ]
            return [self[row_index][index] for row_index in range(len(self))]
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return {
            "image": image,
            "caption": row["caption"],
            "image_id": row.get("image_filename", image_path.name),
            "split": normalize_split(row["split"]),
        }

    def select(self, indices):
        return LocalFlickrSplit(self.rows[index] for index in indices)


def load_local_flickr30k(annotations_path=DEFAULT_LOCAL_ANNOTATIONS):
    annotations_path = Path(annotations_path)
    with annotations_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault(normalize_split(row["split"]), []).append(row)
    return {
        split: LocalFlickrSplit(split_rows)
        for split, split_rows in grouped.items()
    }


def load_flickr30k(dataset_name=DEFAULT_DATASET, cache_dir=None):
    local_annotations = (
        Path(cache_dir) / "annotations.csv"
        if cache_dir and Path(cache_dir).is_dir()
        else DEFAULT_LOCAL_ANNOTATIONS
    )
    if local_annotations.is_file():
        return load_local_flickr30k(local_annotations)
    from datasets import load_dataset

    return load_dataset(dataset_name, cache_dir=cache_dir)


def select_split(dataset_dict, split):
    aliases = SPLIT_ALIASES.get(split, (split,))
    physical_splits = list(dataset_dict.keys())
    physical = next((name for name in aliases if name in dataset_dict), None)
    dataset = dataset_dict[physical] if physical else None
    if dataset is None and len(physical_splits) == 1:
        dataset = dataset_dict[physical_splits[0]]
    if dataset is None:
        raise ValueError(
            f"Split {split!r} was not found. Available splits: {physical_splits}"
        )

    if "split" not in dataset.column_names:
        return dataset
    values = dataset["split"]
    internal = next((name for name in aliases if name in set(values)), None)
    if internal is None:
        if len(physical_splits) == 1:
            raise ValueError(
                f"Split {split!r} was not found in the dataset's 'split' column."
            )
        return dataset
    indices = [index for index, value in enumerate(values) if value == internal]
    return dataset.select(indices)


def first_caption(value):
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value or "")


class Flickr30kDataset(Dataset):
    """Image-caption pairs from the HuggingFace Flickr30k dataset."""

    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = normalize_split(split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample["image"].convert("RGB")
        caption = first_caption(sample.get("caption", sample.get("captions", "")))
        image_id = next(
            (
                sample.get(key)
                for key in ("filename", "file_name", "img_id", "image_id")
                if sample.get(key) is not None
            ),
            index,
        )
        return {
            "image": image,
            "caption": caption,
            "image_id": str(image_id),
            "index": index,
        }


class LocalImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = [Path(path) for path in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as image:
            return {"image": image.convert("RGB"), "image_id": path.name, "index": index}


def make_collate_fn(processor, include_text=True):
    def collate(samples):
        images = [sample["image"] for sample in samples]
        captions = [sample.get("caption", "") for sample in samples]
        encoded = processor(
            images=images,
            text=captions if include_text else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = {"pixel_values": encoded["pixel_values"]}
        if include_text:
            batch["input_ids"] = encoded["input_ids"]
            batch["attention_mask"] = encoded["attention_mask"]
            batch["captions"] = captions
        batch["image_ids"] = [sample["image_id"] for sample in samples]
        batch["indices"] = torch.tensor([sample["index"] for sample in samples])
        return batch

    return collate


def create_dataloader(
    dataset,
    processor,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    include_text=True,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=make_collate_fn(processor, include_text=include_text),
    )


def create_flickr30k_loaders(
    processor,
    dataset_name=DEFAULT_DATASET,
    cache_dir=None,
    splits=("train", "val", "test"),
    batch_size=64,
    num_workers=4,
):
    dataset_dict = load_flickr30k(dataset_name, cache_dir)
    datasets = {
        normalize_split(split): Flickr30kDataset(
            select_split(dataset_dict, split), split
        )
        for split in splits
    }
    loaders = {
        split: create_dataloader(
            dataset,
            processor,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders
