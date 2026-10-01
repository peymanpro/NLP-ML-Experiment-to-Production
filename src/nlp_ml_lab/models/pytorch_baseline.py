from typing import cast

import torch
from torch import nn


class MeanEmbeddingClassifier(nn.Module):
    """Small neural baseline that averages token embeddings before classification."""

    def __init__(
        self,
        vocabulary_size: int,
        num_classes: int,
        *,
        embedding_dim: int = 64,
        pad_id: int = 0,
    ) -> None:
        super().__init__()

        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(dtype=embedded.dtype)
        lengths = mask.sum(dim=1).clamp_min(1.0)
        pooled = (embedded * mask).sum(dim=1) / lengths
        return cast(torch.Tensor, self.classifier(pooled))
