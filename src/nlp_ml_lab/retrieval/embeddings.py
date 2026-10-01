from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SentenceTransformerEmbedder:
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        self.model = SentenceTransformer(self.model_id, device=self.device)

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
