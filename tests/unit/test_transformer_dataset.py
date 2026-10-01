from unittest.mock import Mock

import torch

from nlp_ml_lab.models.transformer_dataset import TransformerTextClassificationDataset


def test_transformer_dataset_returns_tensor_examples() -> None:
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2], [3, 0]]),
        "attention_mask": torch.tensor([[1, 1], [1, 0]]),
    }

    dataset = TransformerTextClassificationDataset(
        ["hello", "world"],
        [0, 1],
        tokenizer=tokenizer,
        max_length=16,
    )

    example = dataset[1]

    assert len(dataset) == 2
    assert example["input_ids"].tolist() == [3, 0]
    assert example["attention_mask"].tolist() == [1, 0]
    assert example["labels"].item() == 1
