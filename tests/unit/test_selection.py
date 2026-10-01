from pathlib import Path

from nlp_ml_lab.models.pytorch_baseline import MeanEmbeddingClassifier
from nlp_ml_lab.training.selection import BestCheckpoint


def test_best_checkpoint_only_updates_on_improvement(tmp_path: Path) -> None:
    model = MeanEmbeddingClassifier(
        vocabulary_size=12,
        num_classes=2,
        embedding_dim=8,
    )
    selection = BestCheckpoint()

    assert selection.consider(
        model=model,
        validation_loss=0.8,
        epoch=1,
        directory=tmp_path,
        metadata={"model": "test"},
    )
    assert selection.best_loss == 0.8
    assert selection.path == tmp_path / "best-model.pt"

    assert not selection.consider(
        model=model,
        validation_loss=0.9,
        epoch=2,
        directory=tmp_path,
        metadata={"model": "test"},
    )
    assert selection.epoch == 1

    assert selection.consider(
        model=model,
        validation_loss=0.5,
        epoch=3,
        directory=tmp_path,
        metadata={"model": "test"},
    )
    assert selection.best_loss == 0.5
    assert selection.epoch == 3
