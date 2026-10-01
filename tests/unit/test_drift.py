import pytest

from nlp_ml_lab.monitoring.drift import (
    categorical_distribution,
    jensen_shannon_divergence,
)


def test_categorical_distribution_sums_to_one() -> None:
    distribution = categorical_distribution(["a", "a", "b"])

    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution["a"] == pytest.approx(2 / 3)


def test_identical_distributions_have_zero_js_divergence() -> None:
    result = jensen_shannon_divergence(["a", "b"], ["a", "b"])

    assert result == pytest.approx(0.0)


def test_shifted_distributions_have_positive_divergence() -> None:
    result = jensen_shannon_divergence(["a", "a"], ["b", "b"])

    assert result > 0
