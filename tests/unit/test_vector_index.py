from unittest.mock import Mock, patch

import numpy as np
import pytest

from nlp_ml_lab.retrieval.vector_index import FaissInnerProductIndex


def test_faiss_index_adds_and_searches_vectors() -> None:
    fake_index = Mock()
    fake_index.ntotal = 2
    fake_index.search.return_value = (
        np.asarray([[0.9, 0.2]], dtype=np.float32),
        np.asarray([[1, 0]], dtype=np.int64),
    )

    with patch(
        "nlp_ml_lab.retrieval.vector_index.faiss",
    ) as faiss_module:
        faiss_module.IndexFlatIP.return_value = fake_index
        index = FaissInnerProductIndex(4)
        index.add(np.ones((2, 4), dtype=np.float32))
        result = index.search(np.ones((1, 4), dtype=np.float32), k=2)

    assert index.size == 2
    assert result[0].ids == [1, 0]
    assert result[0].scores == pytest.approx([0.9, 0.2])


def test_faiss_index_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError):
        FaissInnerProductIndex(0)
