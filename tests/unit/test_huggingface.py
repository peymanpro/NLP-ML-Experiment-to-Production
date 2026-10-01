from unittest.mock import Mock, patch

import pytest

from nlp_ml_lab.models.huggingface import load_sequence_classifier


def test_huggingface_factory_loads_tokenizer_and_model() -> None:
    tokenizer = Mock()
    model = Mock()

    with (
        patch(
            "nlp_ml_lab.models.huggingface.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
        patch(
            "nlp_ml_lab.models.huggingface.AutoModelForSequenceClassification.from_pretrained",
            return_value=model,
        ),
    ):
        bundle = load_sequence_classifier("example/model", num_labels=77)

    assert bundle.tokenizer is tokenizer
    assert bundle.model is model


@pytest.mark.parametrize(
    ("model_id", "num_labels"),
    [("", 77), ("example/model", 1)],
)
def test_huggingface_factory_rejects_invalid_configuration(
    model_id: str,
    num_labels: int,
) -> None:
    with pytest.raises(ValueError):
        load_sequence_classifier(model_id, num_labels=num_labels)
