from dataclasses import dataclass

from nlp_ml_lab.retrieval.dense import DenseRetriever
from nlp_ml_lab.retrieval.embeddings import SentenceTransformerEmbedder


@dataclass
class BiEncoderRetriever:
    document_ids: list[str]
    documents: list[str]
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        self._retriever = DenseRetriever(
            self.document_ids,
            self.documents,
            SentenceTransformerEmbedder(self.model_id),
        )

    def search(self, query: str, *, k: int = 10) -> list[tuple[str, float]]:
        return self._retriever.search(query, k=k)
