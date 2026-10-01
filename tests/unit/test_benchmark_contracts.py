
import pandas as pd
import pytest

from nlp_ml_lab.retrieval.benchmark import (
    load_corpus_frame,
    load_qrels_frame,
    load_queries_frame,
)


def test_benchmark_frames_become_typed_contracts() -> None:
    corpus = load_corpus_frame(
        pd.DataFrame({"_id": ["d1"], "text": ["document"]})
    )
    queries = load_queries_frame(
        pd.DataFrame({"_id": ["q1"], "text": ["query"]})
    )
    qrels = load_qrels_frame(
        pd.DataFrame(
            {"query-id": ["q1"], "corpus-id": ["d1"], "score": [1]}
        )
    )

    assert corpus.document_ids == ["d1"]
    assert queries.query_ids == ["q1"]
    assert qrels.query_to_documents == {"q1": {"d1"}}


def test_benchmark_contract_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        load_corpus_frame(pd.DataFrame({"text": ["document"]}))
