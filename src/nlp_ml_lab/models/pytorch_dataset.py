from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from nlp_ml_lab.text.vocabulary import Vocabulary


@dataclass(frozen=True)
class EncodedExample:
    token_ids: list[int]
    label: int


class TextClassificationDataset(Dataset[EncodedExample]):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        *,
        vocabulary: Vocabulary,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("dataset must not be empty")

        self.examples = [
            EncodedExample(vocabulary.encode(text), int(label))
            for text, label in zip(texts, labels)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]


def collate_text_classification(
    batch: list[EncodedExample],
    *,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("batch must not be empty")

    max_length = max(len(item.token_ids) for item in batch)
    input_ids = torch.full(
        (len(batch), max_length),
        fill_value=pad_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.tensor([item.label for item in batch], dtype=torch.long)

    for row, item in enumerate(batch):
        length = len(item.token_ids)
        if length == 0:
            continue
        input_ids[row, :length] = torch.tensor(item.token_ids, dtype=torch.long)
        attention_mask[row, :length] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
