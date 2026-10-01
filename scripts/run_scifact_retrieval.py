
import argparse
from pathlib import Path

import pandas as pd

from nlp_ml_lab.retrieval.benchmark import (
    load_corpus_frame,
    load_qrels_frame,
    load_queries_frame,
)
from nlp_ml_lab.retrieval.embeddings import SentenceTransformerEmbedder
from nlp_ml_lab.retrieval.metrics import evaluate_many


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dense retrieval on the BEIR SciFact benchmark."
    )
    parser.add_argument(
        "--model-id",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--max-documents", type=int, default=5183)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/scifact-dense.json"),
    )
    return parser.parse_args()


def _limit(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("limit must be positive")
    return frame.head(min(limit, len(frame))).copy()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    corpus = load_dataset(
        "BeIR/scifact",
        "corpus",
        split="corpus",
        cache_dir=str(args.cache_dir),
    ).to_pandas()
    queries = load_dataset(
        "BeIR/scifact",
        "queries",
        split="queries",
        cache_dir=str(args.cache_dir),
    ).to_pandas()
    qrels = load_dataset(
        "BeIR/scifact-qrels",
        split="test",
        cache_dir=str(args.cache_dir),
    ).to_pandas()

    corpus = _limit(corpus, args.max_documents)
    queries = _limit(queries, args.max_queries)

    corpus_contract = load_corpus_frame(corpus)
    query_contract = load_queries_frame(queries)
    qrel_contract = load_qrels_frame(qrels)

    embedder = SentenceTransformerEmbedder(args.model_id)
    doc_embeddings = embedder.encode(corpus_contract.texts)
    query_embeddings = embedder.encode(query_contract.texts)

    scores = query_embeddings @ doc_embeddings.T

    cases = []
    for row_index, query_id in enumerate(query_contract.query_ids):
        relevant = qrel_contract.query_to_documents.get(query_id, set())
        if not relevant:
            continue
        ranked_indices = scores[row_index].argsort()[::-1][: args.top_k]
        ranked_ids = [
            corpus_contract.document_ids[index]
            for index in ranked_indices
        ]
        cases.append((ranked_ids, relevant))

    metrics = evaluate_many(cases, k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        __import__("json").dumps(
            {
                "dataset": "BeIR/scifact",
                "model_id": args.model_id,
                "documents": len(corpus_contract.document_ids),
                "queries": len(cases),
                "top_k": args.top_k,
                "metrics": __import__("dataclasses").asdict(metrics),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
