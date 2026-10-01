from pathlib import Path

import pytest

from nlp_ml_lab.data.metadata import DatasetMetadata, sha256_file


def test_metadata_serializes_to_plain_dictionary() -> None:
    metadata = DatasetMetadata(
        name="example",
        version="1",
        source="https://example.test/dataset",
        license="test-license",
        task="classification",
        description="small test dataset",
    )

    assert metadata.as_dict()["task"] == "classification"


def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")

    first = sha256_file(path)
    second = sha256_file(path)

    assert first == second
    assert len(first) == 64


def test_sha256_file_rejects_invalid_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError):
        sha256_file(path, chunk_size=0)
