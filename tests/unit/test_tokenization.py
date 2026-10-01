from unittest.mock import Mock

import pytest

from nlp_ml_lab.models.tokenization import tokenize_texts


def test_tokenize_texts_uses_bounded_sequence_length() -> None:
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": [[1, 2], [3]],
        "attention_mask": [[1, 1], [1]],
    }

    result = tokenize_texts(
        tokenizer,
        ["hello", "world"],
        max_length=32,
    )

    tokenizer.assert_called_once_with(
        ["hello", "world"],
        truncation=True,
        max_length=32,
        padding=False,
    )
    assert result["input_ids"][0] == [1, 2]


def test_tokenize_texts_rejects_invalid_input() -> None:
    tokenizer = Mock()

    with pytest.raises(ValueError):
        tokenize_texts(tokenizer, [], max_length=32)

    with pytest.raises(ValueError):
        tokenize_texts(tokenizer, ["hello"], max_length=0)
