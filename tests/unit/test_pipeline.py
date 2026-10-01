from unittest.mock import Mock

from nlp_ml_lab.retrieval.lexical import LexicalResult
from nlp_ml_lab.retrieval.pipeline import HybridSearchPipeline


def test_pipeline_fuses_lexical_and_dense_results() -> None:
    lexical = Mock()
    lexical.search.return_value = [
        LexicalResult("d1", 1.0),
        LexicalResult("d2", 0.5),
    ]

    dense = Mock()
    dense.search.return_value = [
        ("d2", 0.9),
        ("d3", 0.8),
    ]

    pipeline = HybridSearchPipeline(
        document_ids=["d1", "d2", "d3"],
        documents=["one", "two", "three"],
        lexical=lexical,
        dense_search=dense,
    )

    result = pipeline.search("query", candidate_k=3, result_k=3)

    assert result[0][0] == "d2"
    assert {item[0] for item in result} == {"d1", "d2", "d3"}
