import pytest

from nlp_ml_lab.training.fine_tuning import FineTuningConfig


def test_fine_tuning_config_accepts_defaults() -> None:
    config = FineTuningConfig()

    assert config.epochs == 3
    assert config.learning_rate == 2e-5


@pytest.mark.parametrize(
    "field",
    ["epochs", "batch_size", "learning_rate", "weight_decay", "max_length"],
)
def test_fine_tuning_config_rejects_invalid_values(field: str) -> None:
    kwargs = {field: -1}
    if field in {"epochs", "batch_size", "learning_rate", "max_length"}:
        kwargs[field] = 0

    with pytest.raises(ValueError):
        FineTuningConfig(**kwargs)
