from nlp_ml_lab.evaluation.latency import summarize_latencies
from nlp_ml_lab.retrieval.hybrid import reciprocal_rank_fusion
from nlp_ml_lab.retrieval.lexical import BM25Retriever
from nlp_ml_lab.retrieval.metrics import evaluate_one
from nlp_ml_lab.text.normalize import normalize_text


def main() -> None:
    documents = [
        ("d1", "card delivery is late"),
        ("d2", "cash withdrawal failed"),
        ("d3", "cash withdrawal fee"),
    ]
    retriever = BM25Retriever(
        [item[0] for item in documents],
        [item[1] for item in documents],
    )

    query = normalize_text("My card delivery is late")
    lexical = [item.document_id for item in retriever.search(query, k=3)]
    fused = reciprocal_rank_fusion([lexical, list(reversed(lexical))], limit=3)
    metrics = evaluate_one(fused, {"d1"}, k=3)
    latency = summarize_latencies([1.0, 2.0, 3.0])

    assert metrics.recall_at_k == 1.0
    assert latency.count == 3
    print("smoke pipeline: OK")
    print(f"retrieved={fused}")
    print(f"recall@3={metrics.recall_at_k:.3f}")
    print(f"mean_latency_ms={latency.mean_ms:.3f}")


if __name__ == "__main__":
    main()
