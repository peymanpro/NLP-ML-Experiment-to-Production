from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def evaluate_one(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    *,
    k: int,
) -> RetrievalMetrics:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0)

    ranked = retrieved_ids[:k]
    hits = [item in relevant_ids for item in ranked]

    recall = len(set(ranked).intersection(relevant_ids)) / len(relevant_ids)
    precision = sum(hits) / k

    reciprocal_rank = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        (1.0 if hit else 0.0) / log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    ndcg = dcg / idcg if idcg else 0.0

    return RetrievalMetrics(
        recall_at_k=float(recall),
        precision_at_k=float(precision),
        reciprocal_rank=float(reciprocal_rank),
        ndcg_at_k=float(ndcg),
    )


def evaluate_many(
    cases: list[tuple[list[str], set[str]]],
    *,
    k: int,
) -> RetrievalMetrics:
    if not cases:
        raise ValueError("cases must not be empty")

    metrics = [
        evaluate_one(retrieved, relevant, k=k)
        for retrieved, relevant in cases
    ]
    return RetrievalMetrics(
        recall_at_k=sum(item.recall_at_k for item in metrics) / len(metrics),
        precision_at_k=sum(item.precision_at_k for item in metrics) / len(metrics),
        reciprocal_rank=sum(item.reciprocal_rank for item in metrics) / len(metrics),
        ndcg_at_k=sum(item.ndcg_at_k for item in metrics) / len(metrics),
    )
