import pytest

from nlp_ml_lab.observability.metrics import InferenceMetrics


def test_inference_metrics_track_requests_errors_and_latency() -> None:
    metrics = InferenceMetrics()
    metrics.observe(10.0)
    metrics.observe(20.0, error=True)

    assert metrics.requests_total == 2
    assert metrics.errors_total == 1
    assert metrics.average_latency_ms == pytest.approx(15.0)


def test_metrics_reject_negative_latency() -> None:
    with pytest.raises(ValueError):
        InferenceMetrics().observe(-1)
