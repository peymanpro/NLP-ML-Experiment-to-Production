from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nlp_ml_lab.data.loaders import load_huggingface_frame


def test_huggingface_frame_loader_converts_split_to_dataframe(tmp_path: Path) -> None:
    fake_dataset = Mock()
    fake_dataset.to_pandas.return_value = pd.DataFrame(
        {"text": ["hello"], "label": [1]}
    )

    with patch(
        "nlp_ml_lab.data.loaders.load_dataset",
        return_value=fake_dataset,
    ) as loader:
        frame = load_huggingface_frame(
            "PolyAI/banking77",
            cache_dir=tmp_path,
            split="train",
        )

    loader.assert_called_once_with(
        "PolyAI/banking77",
        split="train",
        cache_dir=str(tmp_path),
    )
    assert list(frame.columns) == ["text", "label"]


@pytest.mark.parametrize(
    ("dataset_id", "split"),
    [("", "train"), ("PolyAI/banking77", "")],
)
def test_loader_rejects_empty_identifiers(
    tmp_path: Path,
    dataset_id: str,
    split: str,
) -> None:
    with pytest.raises(ValueError):
        load_huggingface_frame(
            dataset_id,
            cache_dir=tmp_path,
            split=split,
        )
