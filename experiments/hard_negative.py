"""Evaluate retrieval with semantically similar caption negatives."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset
from transformers import CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loglo.data import create_dataloader, load_flickr30k, select_split, Flickr30kDataset
from loglo.engine import extract_embeddings, load_checkpoint
from loglo.metrics import hard_negative_metrics
from loglo.models import MODEL_REGISTRY, build_model
from loglo.utils import ensure_dir, format_model_name, get_device, save_csv, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_types", nargs="+", default=("clip", "weighted_sum_cls"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--linear_checkpoint", type=Path)
    parser.add_argument("--patch_checkpoint", type=Path)
    parser.add_argument("--dataset_name", default="nlphuji/flickr30k")
    parser.add_argument("--dataset_cache", type=Path)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--layers", type=int, nargs="+", default=(3, 6, 9, 12))
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_negatives", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result_dir", type=Path, default=Path("results/hard_negative"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def checkpoint_for(model_type, args):
    return {
        "weighted_sum_cls": args.checkpoint,
        "linear_l12": args.linear_checkpoint,
        "weighted_sum_patch_mean": args.patch_checkpoint,
    }.get(model_type)


def mine_negatives(text_embeddings, image_ids, count):
    similarity = text_embeddings @ text_embeddings.T
    same_image = torch.tensor(
        [[left == right for right in image_ids] for left in image_ids],
        dtype=torch.bool,
    )
    similarity.masked_fill_(same_image, -torch.inf)
    if count >= similarity.size(1) or int((~same_image).sum(dim=1).min()) < count:
        raise ValueError("Not enough captions from distinct images for hard negatives.")
    return similarity.topk(count, dim=1).indices


def evaluate_candidates(model_name, images, texts, negatives, captions, image_ids):
    ranks, margins, cases = [], [], []
    for index in range(images.size(0)):
        candidates = torch.cat((torch.tensor([index]), negatives[index]))
        scores = images[index] @ texts[candidates].T
        rank = int((scores[1:] >= scores[0]).sum()) + 1
        negative_offset = int(scores[1:].argmax()) + 1
        negative_index = int(candidates[negative_offset])
        margin = float(scores[0] - scores[negative_offset])
        ranks.append(rank)
        margins.append(margin)
        cases.append(
            {
                "model": model_name,
                "image_id": image_ids[index],
                "positive_caption": captions[index],
                "positive_rank": rank,
                "positive_score": float(scores[0]),
                "top_negative_caption": captions[negative_index],
                "top_negative_score": float(scores[negative_offset]),
                "margin": margin,
                "negative_captions": json.dumps(
                    [captions[int(item)] for item in negatives[index]],
                    ensure_ascii=False,
                ),
            }
        )
    return hard_negative_metrics(ranks, margins), cases


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
    dataset = Flickr30kDataset(raw, args.split)
    count = min(args.num_samples, len(dataset))
    indices = sorted(random.Random(args.seed).sample(range(len(dataset)), count))
    subset = Subset(dataset, indices)
    loader = create_dataloader(
        subset, processor, args.batch_size, False, args.num_workers
    )
    result_rows, case_rows = [], []
    negatives = None
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
        embeddings = extract_embeddings(model, loader, device)
        if negatives is None:
            negatives = mine_negatives(
                embeddings["text_embeddings"],
                embeddings["image_ids"],
                args.num_negatives,
            )
        label = format_model_name(model_type, args.layers)
        metrics, cases = evaluate_candidates(
            label,
            embeddings["image_embeddings"],
            embeddings["text_embeddings"],
            negatives,
            embeddings["captions"],
            embeddings["image_ids"],
        )
        result_rows.append(
            {
                "model": label,
                "num_samples": count,
                "num_negatives": args.num_negatives,
                **metrics,
            }
        )
        case_rows.extend(cases)
    save_csv(result_rows, args.result_dir / "hard_negative_results.csv")
    save_csv(case_rows, args.result_dir / "hard_negative_cases.csv")
    print(result_rows)


if __name__ == "__main__":
    main()
