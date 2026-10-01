from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k_constant: int = 60,
    limit: int = 10,
) -> list[str]:
    if not ranked_lists:
        return []
    if k_constant <= 0:
        raise ValueError("k_constant must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")

    scores: dict[str, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] += 1.0 / (k_constant + rank)

    return [
        document_id
        for document_id, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]
