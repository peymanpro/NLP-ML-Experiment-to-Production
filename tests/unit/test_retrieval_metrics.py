import pytest

from nlp_ml_lab.retrieval.metrics import evaluate_one


def test_retrieval_metrics_capture_rank_and_recall() -> None:
    metrics = evaluate_one(
        ["d2", "d1", "d3"],
        {"d1", "d4"},
        k=3,
    )

    assert metrics.recall_at_k == pytest.approx(0.5)
    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank == pytest.approx(0.5)
    assert metrics.ndcg_at_k > 0


def test_retrieval_metrics_handle_no_relevant_ids() -> None:
    metrics = evaluate_one(["d1"], set(), k=5)

    assert metrics.recall_at_k == 0
    assert metrics.precision_at_k == 0
