import pytest

from nlp_ml_lab.evaluation.latency import summarize_latencies


def test_latency_summary_reports_basic_percentiles() -> None:
    result = summarize_latencies([1, 2, 3, 4, 5])

    assert result.count == 5
    assert result.mean_ms == pytest.approx(3.0)
    assert result.p50_ms == 3.0
    assert result.p95_ms == 5.0
