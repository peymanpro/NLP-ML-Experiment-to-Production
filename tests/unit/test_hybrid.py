from nlp_ml_lab.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_merges_ranked_lists_deterministically() -> None:
    result = reciprocal_rank_fusion(
        [["a", "b", "c"], ["c", "b", "d"]],
        k_constant=1,
        limit=4,
    )

    assert result[:2] == ["c", "b"]


def test_rrf_respects_limit() -> None:
    result = reciprocal_rank_fusion(
        [["a", "b", "c"]],
        limit=2,
    )

    assert result == ["a", "b"]
