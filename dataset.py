import csv
import json
import os
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


FLICKR30K_DATASET_NAME = "nlphuji/flickr30k"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "flickr30k"
DEFAULT_IMAGE_DIR = DEFAULT_DATA_DIR / "images"
DEFAULT_ANNOTATIONS_PATH = DEFAULT_DATA_DIR / "annotations.csv"
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "validation"),
    "validation": ("validation", "val"),
    "test": ("test",),
}


def normalize_split(split):
    if split == "validation":
        return "val"
    return split


def load_huggingface_flickr30k(dataset_name=FLICKR30K_DATASET_NAME):
    from datasets import load_dataset

    return load_dataset(dataset_name)


def get_huggingface_split(dataset_dict, split):
    physical_splits = list(dataset_dict.keys())
    aliases = SPLIT_ALIASES.get(split, (split,))

    if len(physical_splits) == 1:
        dataset = dataset_dict[physical_splits[0]]
    else:
        physical_match = next(
            (name for name in aliases if name in dataset_dict),
            None,
        )
        if physical_match is None:
            raise ValueError(
                f"Unknown Flickr30k split {split!r}. "
                f"Available dataset splits: {physical_splits}"
            )
        dataset = dataset_dict[physical_match]

    if "split" not in dataset.column_names:
        return dataset

    internal_values = dataset["split"]
    internal_splits = sorted(set(internal_values))
    internal_match = next(
        (name for name in aliases if name in internal_splits),
        None,
    )
    if internal_match is None:
        if len(physical_splits) == 1:
            raise ValueError(
                f"Unknown Flickr30k split {split!r}. "
                f"Available internal splits: {internal_splits}"
            )
        return dataset

    indices = [
        index
        for index, sample_split in enumerate(internal_values)
        if sample_split == internal_match
    ]
    return dataset.select(indices)


def select_caption(caption):
    if isinstance(caption, (list, tuple)):
        return str(caption[0]) if caption else ""
    return str(caption or "")


def _safe_stem(value):
    stem = Path(str(value)).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")


def _image_filename(sample, split, index):
    for key in ("filename", "file_name", "img_id", "image_id"):
        if sample.get(key) is not None:
            stem = _safe_stem(sample[key])
            if stem:
                return f"{index:06d}_{stem}.jpg"
    return f"{index:06d}.jpg"


def prepare_flickr30k(
    data_dir=DEFAULT_DATA_DIR,
    annotations_path=None,
    dataset_name=FLICKR30K_DATASET_NAME,
    splits=("train", "val", "test"),
):
    data_dir = Path(data_dir).resolve()
    image_dir = data_dir / "images"
    annotations_path = (
        Path(annotations_path).resolve()
        if annotations_path
        else data_dir / "annotations.csv"
    )
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HuggingFace dataset: {dataset_name}")
    dataset_dict = load_huggingface_flickr30k(dataset_name)
    records = []
    for requested_split in splits:
        split = normalize_split(requested_split)
        dataset = get_huggingface_split(dataset_dict, requested_split)
        split_dir = image_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        progress = tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc=f"Preparing {split}",
            unit="image",
        )
        for index, sample in progress:
            image_path = split_dir / _image_filename(sample, split, index)
            if not image_path.is_file():
                sample["image"].convert("RGB").save(
                    image_path,
                    format="JPEG",
                    quality=95,
                )

            try:
                stored_path = image_path.relative_to(PROJECT_ROOT)
            except ValueError:
                stored_path = image_path
            records.append(
                {
                    "split_index": index,
                    "image_filename": image_path.name,
                    "image_path": stored_path.as_posix(),
                    "caption": select_caption(sample.get("caption", "")),
                    "split": split,
                }
            )

    write_annotations(records, annotations_path)
    return records


def write_annotations(records, annotations_path):
    annotations_path = Path(annotations_path)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    if annotations_path.suffix.lower() == ".json":
        with annotations_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
        return
    if annotations_path.suffix.lower() != ".csv":
        raise ValueError("Annotations path must end in .csv or .json.")

    with annotations_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = (
            "split_index",
            "image_filename",
            "image_path",
            "caption",
            "split",
        )
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


def load_annotations(annotations_path=DEFAULT_ANNOTATIONS_PATH, split=None):
    annotations_path = Path(annotations_path).resolve()
    if not annotations_path.is_file():
        raise FileNotFoundError(
            f"Annotations not found: {annotations_path}. "
            "Run `python prepare_flickr30k.py` first."
        )

    if annotations_path.suffix.lower() == ".json":
        with annotations_path.open(encoding="utf-8") as file:
            records = json.load(file)
    elif annotations_path.suffix.lower() == ".csv":
        with annotations_path.open(encoding="utf-8", newline="") as file:
            records = list(csv.DictReader(file))
    else:
        raise ValueError("Annotations path must end in .csv or .json.")

    split_counts = {}
    for record in records:
        record.setdefault("image_filename", Path(record["image_path"]).name)
        normalized_split = normalize_split(record["split"])
        record.setdefault("split_index", split_counts.get(normalized_split, 0))
        split_counts[normalized_split] = split_counts.get(normalized_split, 0) + 1

    if split is None:
        return records
    normalized = normalize_split(split)
    return [
        record
        for record in records
        if normalize_split(record["split"]) == normalized
    ]


def resolve_image_path(image_path, annotations_path=DEFAULT_ANNOTATIONS_PATH):
    image_path = Path(image_path).expanduser()
    if image_path.is_absolute():
        return image_path

    annotations_path = Path(annotations_path).resolve()
    candidates = (
        PROJECT_ROOT / image_path,
        annotations_path.parent / image_path,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


class Flickr30kDataset(Dataset):
    def __init__(
        self,
        processor,
        split="train",
        annotations_path=DEFAULT_ANNOTATIONS_PATH,
    ):
        self.processor = processor
        self.split = normalize_split(split)
        self.annotations_path = Path(annotations_path).resolve()
        self.records = load_annotations(self.annotations_path, self.split)
        if not self.records:
            raise ValueError(
                f"No samples found for split {self.split!r} in "
                f"{self.annotations_path}."
            )

    def __len__(self):
        return len(self.records)

    def get_record(self, index):
        record = dict(self.records[index])
        record["image_path"] = resolve_image_path(
            record["image_path"],
            self.annotations_path,
        )
        return record

    def __getitem__(self, index):
        record = self.get_record(index)
        with Image.open(record["image_path"]) as image:
            pixel_values = self.processor(
                images=image.convert("RGB"),
                return_tensors="pt",
            )["pixel_values"].squeeze(0)
        return pixel_values, record["caption"]


def create_dataloader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=None,
    pin_memory=None,
):
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    if pin_memory is None:
        import torch

        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
