from dataclasses import dataclass

from sentence_transformers import CrossEncoder


@dataclass
class CrossEncoderReranker:
    model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    device: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        self.model = CrossEncoder(self.model_id, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        *,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if k <= 0:
            raise ValueError("k must be positive")
        if not candidates:
            return []

        pairs = [(query, document) for _, document in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: (-float(item[1]), item[0][0]),
        )
        return [(document_id, float(score)) for (document_id, _), score in ranked[:k]]
