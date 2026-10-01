from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from nlp_ml_lab.serving.loader import load_predictor_from_environment


def test_model_loader_requires_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NLP_ML_MODEL_PATH", raising=False)

    with pytest.raises(RuntimeError):
        load_predictor_from_environment()


def test_model_loader_reads_local_model_and_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "labels.json").write_text('["first", "second"]', encoding="utf-8")
    monkeypatch.setenv("NLP_ML_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("NLP_ML_MODEL_VERSION", "2026.10")

    fake_tokenizer = Mock()
    fake_model = Mock()
    fake_model.config.num_labels = 2

    with (
        patch(
            "nlp_ml_lab.serving.loader.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ),
        patch(
            "nlp_ml_lab.serving.loader.AutoModelForSequenceClassification.from_pretrained",
            return_value=fake_model,
        ),
        patch("nlp_ml_lab.serving.loader.torch.device", return_value=torch.device("cpu")),
    ):
        predictor = load_predictor_from_environment()

    assert predictor.model_version == "2026.10"
    assert predictor.labels == ["first", "second"]
