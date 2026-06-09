import argparse
from pathlib import Path

from dataset import (
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_DATA_DIR,
    prepare_flickr30k,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save nlphuji/flickr30k images and annotations locally."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Local Flickr30k directory (default: data/flickr30k).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS_PATH,
        help="Output .csv or .json annotations path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = prepare_flickr30k(
        data_dir=args.data_dir,
        annotations_path=args.annotations,
    )
    split_counts = {}
    for record in records:
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1
    print(f"Saved annotations: {args.annotations.resolve()}")
    print(f"Images directory: {(args.data_dir / 'images').resolve()}")
    print(
        "Samples | "
        + " | ".join(
            f"{split}: {count:,}"
            for split, count in split_counts.items()
        )
    )


if __name__ == "__main__":
    main()
