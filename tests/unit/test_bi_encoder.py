from unittest.mock import Mock

from nlp_ml_lab.retrieval.bi_encoder import BiEncoderRetriever


def test_bi_encoder_accepts_injected_embedder() -> None:
    dense = Mock()
    dense.search.return_value = [("d1", 0.8)]
    embedder = Mock()

    from unittest.mock import patch

    with patch(
        "nlp_ml_lab.retrieval.bi_encoder.DenseRetriever",
        return_value=dense,
    ) as constructor:
        retriever = BiEncoderRetriever(
            ["d1"],
            ["document"],
            "example/encoder",
            embedder=embedder,
        )

    assert retriever.search("query") == [("d1", 0.8)]
    constructor.assert_called_once_with(["d1"], ["document"], embedder)
