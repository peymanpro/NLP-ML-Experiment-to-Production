from dataclasses import dataclass

from nlp_ml_lab.retrieval.dense import DenseRetriever
from nlp_ml_lab.retrieval.embeddings import SentenceTransformerEmbedder


@dataclass
class BiEncoderRetriever:
    document_ids: list[str]
    documents: list[str]
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder: SentenceTransformerEmbedder | None = None

    def __post_init__(self) -> None:
        embedder = self.embedder or SentenceTransformerEmbedder(self.model_id)
        self._retriever = DenseRetriever(
            self.document_ids,
            self.documents,
            embedder,
        )

    def search(self, query: str, *, k: int = 10) -> list[tuple[str, float]]:
        return self._retriever.search(query, k=k)
