import pytest

from nlp_ml_lab.models.classical import train_logistic_regression


def test_logistic_regression_baseline_can_learn_tiny_dataset() -> None:
    model = train_logistic_regression(
        [
            "card arrived today",
            "card is late",
            "cash withdrawal failed",
            "atm cash issue",
            "card arrived quickly",
            "cash machine failed",
        ],
        [0, 0, 1, 1, 0, 1],
        random_state=11,
    )

    predictions = model.predict(["my card is late", "cash machine issue"])

    assert len(predictions) == 2
    assert set(predictions) <= {0, 1}


def test_logistic_regression_requires_multiple_classes() -> None:
    with pytest.raises(ValueError):
        train_logistic_regression(["only one class"], [0], random_state=11)
