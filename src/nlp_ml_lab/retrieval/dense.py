from dataclasses import dataclass

from nlp_ml_lab.retrieval.embeddings import SentenceTransformerEmbedder
from nlp_ml_lab.retrieval.vector_index import FaissInnerProductIndex


@dataclass
class DenseRetriever:
    document_ids: list[str]
    documents: list[str]
    embedder: SentenceTransformerEmbedder

    def __post_init__(self) -> None:
        if len(self.document_ids) != len(self.documents):
            raise ValueError("document_ids and documents must have the same length")
        if not self.documents:
            raise ValueError("documents must not be empty")

        vectors = self.embedder.encode(self.documents)
        self.index = FaissInnerProductIndex(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, *, k: int = 10) -> list[tuple[str, float]]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if k <= 0:
            raise ValueError("k must be positive")

        query_vector = self.embedder.encode([query])
        result = self.index.search(query_vector, k=k)[0]

        return [
            (self.document_ids[index], score)
            for index, score in zip(result.ids, result.scores)
        ]
