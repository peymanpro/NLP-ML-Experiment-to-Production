from pathlib import Path

import torch

from nlp_ml_lab.models.pytorch_baseline import MeanEmbeddingClassifier
from nlp_ml_lab.training.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = MeanEmbeddingClassifier(
        vocabulary_size=12,
        num_classes=3,
        embedding_dim=8,
    )
    path = tmp_path / "model.pt"

    save_checkpoint(
        model,
        path=path,
        metadata={"model": "mean-embedding", "version": "0.1"},
    )

    restored = MeanEmbeddingClassifier(
        vocabulary_size=12,
        num_classes=3,
        embedding_dim=8,
    )
    metadata = load_checkpoint(restored, path=path)

    assert metadata["model"] == "mean-embedding"
    for first, second in zip(model.parameters(), restored.parameters()):
        assert torch.equal(first, second)
