import torch
from torch.utils.data import DataLoader

from nlp_ml_lab.models.pytorch_dataset import (
    TextClassificationDataset,
    collate_text_classification,
)
from nlp_ml_lab.models.pytorch_baseline import MeanEmbeddingClassifier
from nlp_ml_lab.text.vocabulary import build_vocabulary
from nlp_ml_lab.training.loops import evaluate, train_one_epoch


def make_loader() -> DataLoader:
    texts = ["card arrived", "card late", "cash failed", "cash issue"] * 3
    labels = [0, 0, 1, 1] * 3
    vocabulary = build_vocabulary(texts)
    dataset = TextClassificationDataset(texts, labels, vocabulary=vocabulary)
    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: collate_text_classification(
            batch,
            pad_id=vocabulary.pad_id,
        ),
    )


def test_training_loop_updates_model_and_returns_result() -> None:
    loader = make_loader()
    model = MeanEmbeddingClassifier(
        vocabulary_size=32,
        num_classes=2,
        embedding_dim=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    before = model.classifier.weight.detach().clone()
    result = train_one_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
    )

    assert result.samples == 12
    assert result.loss > 0
    assert not torch.equal(before, model.classifier.weight.detach())


def test_evaluation_loop_returns_loss() -> None:
    loader = make_loader()
    model = MeanEmbeddingClassifier(
        vocabulary_size=32,
        num_classes=2,
        embedding_dim=8,
    )

    result = evaluate(model, loader, device=torch.device("cpu"))

    assert result.samples == 12
    assert result.loss > 0
