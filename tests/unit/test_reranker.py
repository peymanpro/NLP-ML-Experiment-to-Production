from unittest.mock import Mock, patch

import pytest

from nlp_ml_lab.retrieval.reranker import CrossEncoderReranker


def test_reranker_orders_candidates_by_model_score() -> None:
    fake_model = Mock()
    fake_model.predict.return_value = [0.2, 0.9, 0.5]

    with patch(
        "nlp_ml_lab.retrieval.reranker.CrossEncoder",
        return_value=fake_model,
    ):
        reranker = CrossEncoderReranker("example/reranker")

    result = reranker.rerank(
        "query",
        [("a", "doc a"), ("b", "doc b"), ("c", "doc c")],
    )

    assert result == [("b", pytest.approx(0.9)), ("c", pytest.approx(0.5)), ("a", pytest.approx(0.2))]


def test_reranker_rejects_blank_query() -> None:
    with patch(
        "nlp_ml_lab.retrieval.reranker.CrossEncoder",
        return_value=Mock(),
    ):
        reranker = CrossEncoderReranker("example/reranker")

    with pytest.raises(ValueError):
        reranker.rerank("", [("a", "doc")])
