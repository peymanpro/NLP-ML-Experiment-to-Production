from unittest.mock import Mock, patch

from nlp_ml_lab.retrieval.bi_encoder import BiEncoderRetriever


def test_bi_encoder_builds_dense_retriever() -> None:
    dense = Mock()
    dense.search.return_value = [("d1", 0.8)]

    with patch(
        "nlp_ml_lab.retrieval.bi_encoder.DenseRetriever",
        return_value=dense,
    ) as constructor:
        retriever = BiEncoderRetriever(
            ["d1"],
            ["document"],
            "example/encoder",
        )

    assert retriever.search("query") == [("d1", 0.8)]
    constructor.assert_called_once()
