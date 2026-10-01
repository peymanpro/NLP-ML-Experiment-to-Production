from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_huggingface_frame(
    dataset_id: str,
    *,
    cache_dir: Path,
    split: str,
) -> pd.DataFrame:
    """Load one Hugging Face split into a dataframe.

    The loader is deliberately generic so benchmark-specific code does not
    leak into training and evaluation modules.
    """

    if not dataset_id.strip():
        raise ValueError("dataset_id must not be empty")
    if not split.strip():
        raise ValueError("split must not be empty")

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_id, split=split, cache_dir=str(cache_dir))

    if not hasattr(dataset, "to_pandas"):
        raise TypeError(f"dataset split does not support dataframe conversion: {type(dataset)!r}")

    frame = dataset.to_pandas()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Hugging Face dataset did not produce a pandas DataFrame")

    return frame.reset_index(drop=True)


def load_banking77(
    *,
    cache_dir: Path,
    dataset_id: str = "PolyAI/banking77",
) -> Mapping[str, pd.DataFrame]:
    """Load the available BANKING77 splits without storing the dataset in Git."""

    requested_splits = ("train", "test")
    loaded: dict[str, pd.DataFrame] = {}

    for split in requested_splits:
        loaded[split] = load_huggingface_frame(
            dataset_id,
            cache_dir=cache_dir,
            split=split,
        )

    return loaded
