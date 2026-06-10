"""Evaluate image-text retrieval under deterministic image corruptions."""

import argparse
import random
import sys
from pathlib import Path

from PIL import ImageDraw
from torch.utils.data import Dataset, Subset
from torchvision.transforms import functional as TF
from transformers import CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loglo.data import (
    Flickr30kDataset,
    create_dataloader,
    load_flickr30k,
    select_split,
)
from loglo.engine import evaluate_retrieval, load_checkpoint
from loglo.metrics import flatten_retrieval_metrics, robustness_summary
from loglo.models import MODEL_REGISTRY, build_model
from loglo.utils import ensure_dir, format_model_name, get_device, save_csv, set_seed


DEFAULT_CORRUPTIONS = (
    "original",
    "gaussian_blur",
    "grayscale",
    "color_jitter",
    "center_crop",
    "occlusion",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_types", nargs="+", default=("clip", "weighted_sum_cls", "weighted_sum_patch_mean"))
    parser.add_argument("--checkpoint_weighted", type=Path)
    parser.add_argument("--checkpoint_patch", type=Path)
    parser.add_argument("--dataset_name", default="nlphuji/flickr30k")
    parser.add_argument("--dataset_cache", type=Path)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--layers", type=int, nargs="+", default=(3, 6, 9, 12))
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result_dir", type=Path, default=Path("results/robustness"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def corrupt(image, name):
    if name == "original":
        return image
    if name == "gaussian_blur":
        return TF.gaussian_blur(image, 9, 2.0)
    if name == "grayscale":
        return TF.to_grayscale(image, num_output_channels=3)
    if name == "color_jitter":
        image = TF.adjust_brightness(image, 0.75)
        image = TF.adjust_contrast(image, 1.25)
        image = TF.adjust_saturation(image, 0.5)
        return TF.adjust_hue(image, 0.05)
    if name == "center_crop":
        width, height = image.size
        size = max(1, int(min(width, height) * 0.75))
        return TF.center_crop(image, [size, size])
    if name == "occlusion":
        image = image.copy()
        width, height = image.size
        box_width, box_height = int(width * 0.3), int(height * 0.3)
        left, top = (width - box_width) // 2, (height - box_height) // 2
        ImageDraw.Draw(image).rectangle(
            (left, top, left + box_width, top + box_height), fill=(0, 0, 0)
        )
        return image
    raise ValueError(f"Unknown corruption: {name}")


class CorruptedDataset(Dataset):
    def __init__(self, dataset, corruption):
        self.dataset = dataset
        self.corruption = corruption

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["image"] = corrupt(sample["image"], self.corruption)
        return sample


def checkpoint_for(model_type, args):
    return {
        "weighted_sum_cls": args.checkpoint_weighted,
        "weighted_sum_patch_mean": args.checkpoint_patch,
    }.get(model_type)


def main():
    args = parse_args()
    invalid = set(args.model_types) - set(MODEL_REGISTRY)
    if invalid:
        raise ValueError(f"Unknown model types: {sorted(invalid)}")
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.result_dir)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    raw = select_split(load_flickr30k(args.dataset_name, args.dataset_cache), args.split)
    base = Flickr30kDataset(raw, args.split)
    count = min(args.num_samples, len(base))
    indices = sorted(random.Random(args.seed).sample(range(len(base)), count))
    rows = []
    for model_type in args.model_types:
        model = build_model(
            model_type, args.clip_model, tuple(args.layers), args.alpha
        ).to(device)
        checkpoint = checkpoint_for(model_type, args)
        if model_type != "clip":
            if checkpoint is None or not checkpoint.is_file():
                print(f"Skipping {model_type}: checkpoint not found: {checkpoint}")
                continue
            load_checkpoint(checkpoint, model, map_location=device)
        label = format_model_name(model_type, args.layers)
        for corruption in args.corruptions:
            dataset = CorruptedDataset(Subset(base, indices), corruption)
            loader = create_dataloader(
                dataset, processor, args.batch_size, False, args.num_workers
            )
            metrics, _ = evaluate_retrieval(model, loader, device)
            row = {
                "model": label,
                "corruption": corruption,
                "num_samples": count,
                **flatten_retrieval_metrics(metrics),
            }
            rows.append(row)
            print(row)
    save_csv(rows, args.result_dir / "robustness_results.csv")
    save_csv(
        robustness_summary(rows), args.result_dir / "robustness_summary.csv"
    )


if __name__ == "__main__":
    main()
