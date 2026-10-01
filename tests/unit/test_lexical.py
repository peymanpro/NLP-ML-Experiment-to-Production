from nlp_ml_lab.retrieval.lexical import BM25Retriever


def test_bm25_prefers_matching_document() -> None:
    retriever = BM25Retriever(
        ["d1", "d2"],
        ["card delivery is late", "cash withdrawal failed"],
    )

    result = retriever.search("card delivery", k=2)

    assert result[0].document_id == "d1"
    assert result[0].score > result[1].score


def test_bm25_results_are_deterministic() -> None:
    retriever = BM25Retriever(
        ["d2", "d1"],
        ["same text", "same text"],
    )

    first = [item.document_id for item in retriever.search("same")]
    second = [item.document_id for item in retriever.search("same")]

    assert first == second
