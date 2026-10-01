from dataclasses import dataclass

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only in minimal environments
    faiss = None


@dataclass(frozen=True)
class VectorSearchResult:
    ids: list[int]
    scores: list[float]


class FaissInnerProductIndex:
    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if faiss is None:
            raise RuntimeError("faiss-cpu is required for FaissInnerProductIndex")
        self._index = faiss.IndexFlatIP(dimension)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if matrix.shape[0] == 0:
            raise ValueError("vectors must not be empty")
        self._index.add(matrix)

    def search(self, query_vectors: np.ndarray, *, k: int) -> list[VectorSearchResult]:
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim != 2:
            raise ValueError("query_vectors must be a 2D array")
        if queries.shape[0] == 0:
            raise ValueError("query_vectors must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")

        scores, indices = self._index.search(queries, min(k, self.size))
        return [
            VectorSearchResult(
                ids=[int(value) for value in row_indices if value >= 0],
                scores=[float(value) for value, row_indices in zip(row_scores, row_indices) if row_indices >= 0],
            )
            for row_indices, row_scores in zip(indices, scores)
        ]
