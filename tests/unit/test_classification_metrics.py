import pytest

from nlp_ml_lab.evaluation.classification import classification_metrics


def test_classification_metrics_are_consistent() -> None:
    metrics = classification_metrics([0, 1, 1], [0, 1, 0])

    assert metrics.accuracy == pytest.approx(2 / 3)
    assert 0.0 <= metrics.precision_macro <= 1.0
    assert 0.0 <= metrics.recall_macro <= 1.0
    assert 0.0 <= metrics.f1_macro <= 1.0
