import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nlp_ml_lab.config import ProjectPaths
from nlp_ml_lab.data.loaders import load_banking77
from nlp_ml_lab.data.splits import split_frame
from nlp_ml_lab.evaluation.classification import classification_metrics
from nlp_ml_lab.models.pytorch_baseline import MeanEmbeddingClassifier
from nlp_ml_lab.models.pytorch_dataset import TextClassificationDataset, collate_text_classification
from nlp_ml_lab.reproducibility import seed_everything
from nlp_ml_lab.text.vocabulary import build_vocabulary
from nlp_ml_lab.training.config import TrainingConfig
from nlp_ml_lab.training.loops import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PyTorch BANKING77 baseline.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=2500)
    parser.add_argument("--max-validation", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    return parser.parse_args()


@torch.no_grad()
def predict(
    model: MeanEmbeddingClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
) -> list[int]:
    model.eval()
    predictions: list[int] = []

    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    return predictions


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    seed_everything(config.seed)

    paths = ProjectPaths.from_root(Path.cwd())
    paths.create_runtime_directories()

    loaded = load_banking77(cache_dir=args.cache_dir)
    split = split_frame(
        loaded["train"],
        label_column="label",
        seed=config.seed,
        test_size=0.1,
        validation_size=0.1,
    )

    train_frame = split.train.head(args.max_train).copy()
    validation_frame = split.validation.head(args.max_validation).copy()

    vocabulary = build_vocabulary(train_frame["text"].tolist())
    train_dataset = TextClassificationDataset(
        train_frame["text"].tolist(),
        train_frame["label"].tolist(),
        vocabulary=vocabulary,
    )
    validation_dataset = TextClassificationDataset(
        validation_frame["text"].tolist(),
        validation_frame["label"].tolist(),
        vocabulary=vocabulary,
    )

    collator = lambda batch: collate_text_classification(
        batch,
        pad_id=vocabulary.pad_id,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeanEmbeddingClassifier(
        vocabulary_size=vocabulary.size,
        num_classes=int(loaded["train"]["label"].nunique()),
        embedding_dim=config.embedding_dim,
        pad_id=vocabulary.pad_id,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = []
    for epoch in range(1, config.epochs + 1):
        train_result = train_one_epoch(
            model, train_loader, optimizer, device=device
        )
        validation_result = evaluate(
            model, validation_loader, device=device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "validation_loss": validation_result.loss,
            }
        )

    predictions = predict(model, validation_loader, device=device)
    metrics = classification_metrics(
        validation_frame["label"].tolist(),
        predictions,
    )

    output = paths.reports / "pytorch-baseline.json"
    output.write_text(
        __import__("json").dumps(
            {
                "config": {
                    "seed": config.seed,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "embedding_dim": config.embedding_dim,
                },
                "device": str(device),
                "vocabulary_size": vocabulary.size,
                "train_samples": len(train_frame),
                "validation_samples": len(validation_frame),
                "history": history,
                "metrics": __import__("dataclasses").asdict(metrics),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
