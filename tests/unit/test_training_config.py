import pytest

from nlp_ml_lab.training.config import TrainingConfig


def test_training_config_accepts_defaults() -> None:
    config = TrainingConfig()

    assert config.epochs == 3
    assert config.batch_size == 32


@pytest.mark.parametrize(
    "field",
    ["epochs", "batch_size", "learning_rate", "embedding_dim"],
)
def test_training_config_rejects_non_positive_values(field: str) -> None:
    kwargs = {field: 0}
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)
