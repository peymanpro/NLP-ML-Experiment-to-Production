import pytest
import torch

from nlp_ml_lab.models.pytorch_baseline import MeanEmbeddingClassifier


def test_classifier_returns_class_logits() -> None:
    model = MeanEmbeddingClassifier(
        vocabulary_size=20,
        num_classes=3,
        embedding_dim=8,
    )

    logits = model(
        torch.tensor([[1, 2, 0], [3, 4, 5]]),
        torch.tensor([[1, 1, 0], [1, 1, 1]]),
    )

    assert logits.shape == (2, 3)


@pytest.mark.parametrize(
    ("vocabulary_size", "num_classes", "embedding_dim"),
    [(0, 3, 8), (10, 1, 8), (10, 3, 0)],
)
def test_classifier_rejects_invalid_dimensions(
    vocabulary_size: int,
    num_classes: int,
    embedding_dim: int,
) -> None:
    with pytest.raises(ValueError):
        MeanEmbeddingClassifier(
            vocabulary_size=vocabulary_size,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )
