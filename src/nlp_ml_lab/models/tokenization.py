from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    max_length: int = 128,
) -> dict[str, Any]:
    if not texts:
        raise ValueError("texts must not be empty")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return dict(encoded)
