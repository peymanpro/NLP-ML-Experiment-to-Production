import torch

from nlp_ml_lab.models.pytorch_dataset import (
    TextClassificationDataset,
    collate_text_classification,
)
from nlp_ml_lab.text.vocabulary import build_vocabulary


def test_dataset_and_collator_produce_expected_tensors() -> None:
    texts = ["card arrived", "cash failed"]
    labels = [0, 1]
    vocabulary = build_vocabulary(texts)
    dataset = TextClassificationDataset(texts, labels, vocabulary=vocabulary)

    batch = collate_text_classification(
        [dataset[0], dataset[1]],
        pad_id=vocabulary.pad_id,
    )

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].tolist() == labels
