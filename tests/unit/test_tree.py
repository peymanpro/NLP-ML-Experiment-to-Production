import pytest

from nlp_ml_lab.models.tree import train_random_forest_baseline


def test_random_forest_baseline_can_predict_tiny_dataset() -> None:
    model = train_random_forest_baseline(
        [
            "card arrived today",
            "card is late",
            "cash withdrawal failed",
            "atm cash issue",
            "cash withdrawal problem",
            "my new card arrived",
            "cash machine failed",
            "my card has not arrived",
        ],
        [0, 0, 1, 1, 1, 0, 1, 0],
        random_state=11,
        components=2,
        estimators=10,
    )

    predictions = model.predict(["my card is late", "cash machine issue"])

    assert len(predictions) == 2
    assert set(predictions) <= {0, 1}


@pytest.mark.parametrize(
    ("components", "estimators"),
    [(0, 10), (2, 0)],
)
def test_random_forest_rejects_invalid_configuration(
    components: int,
    estimators: int,
) -> None:
    with pytest.raises(ValueError):
        train_random_forest_baseline(
            ["a", "b"],
            [0, 1],
            random_state=11,
            components=components,
            estimators=estimators,
        )
