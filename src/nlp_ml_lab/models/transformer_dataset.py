from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TransformerExample:
    text: str
    label: int


class TransformerTextClassificationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        *,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("dataset must not be empty")

        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("tokenizer input_ids must be a tensor")
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("tokenizer attention_mask must be a tensor")

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }
