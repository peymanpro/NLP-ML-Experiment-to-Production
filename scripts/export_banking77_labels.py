import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/serving/labels.json"),
    )
    args = parser.parse_args()

    dataset = load_dataset(
        "PolyAI/banking77",
        split="train",
        cache_dir=str(args.cache_dir),
    )

    names = dataset.features["label"].names
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(names, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
