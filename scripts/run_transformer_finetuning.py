import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nlp_ml_lab.config import ProjectPaths
from nlp_ml_lab.data.loaders import load_banking77
from nlp_ml_lab.data.splits import split_frame
from nlp_ml_lab.evaluation.classification import classification_metrics
from nlp_ml_lab.models.huggingface import load_sequence_classifier
from nlp_ml_lab.models.transformer import TransformerClassifier
from nlp_ml_lab.models.transformer_dataset import TransformerTextClassificationDataset
from nlp_ml_lab.reproducibility import seed_everything
from nlp_ml_lab.training.checkpoints import load_checkpoint
from nlp_ml_lab.training.fine_tuning import FineTuningConfig
from nlp_ml_lab.training.loops import evaluate, train_one_epoch
from nlp_ml_lab.training.selection import BestCheckpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer on BANKING77.")
    parser.add_argument("--model-id", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=2500)
    parser.add_argument("--max-validation", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    return parser.parse_args()


@torch.no_grad()
def predict(
    model: TransformerClassifier,
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
    config = FineTuningConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
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

    bundle = load_sequence_classifier(
        args.model_id,
        num_labels=int(loaded["train"]["label"].nunique()),
    )
    model = TransformerClassifier(bundle.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TransformerTextClassificationDataset(
        train_frame["text"].tolist(),
        train_frame["label"].tolist(),
        tokenizer=bundle.tokenizer,
        max_length=config.max_length,
    )
    validation_dataset = TransformerTextClassificationDataset(
        validation_frame["text"].tolist(),
        validation_frame["label"].tolist(),
        tokenizer=bundle.tokenizer,
        max_length=config.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    selection = BestCheckpoint()
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
        )
        validation_result = evaluate(
            model,
            validation_loader,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "validation_loss": validation_result.loss,
            }
        )
        selection.consider(
            model=model,
            validation_loss=validation_result.loss,
            epoch=epoch,
            directory=paths.models,
            metadata={
                "model_id": args.model_id,
                "task": "banking77",
            },
        )

    if selection.path is None:
        raise RuntimeError("fine-tuning completed without a checkpoint")

    metadata = load_checkpoint(
        model,
        path=selection.path,
    )
    predictions = predict(model, validation_loader, device=device)
    metrics = classification_metrics(
        validation_frame["label"].tolist(),
        predictions,
    )

    output = paths.reports / "transformer-finetuning.json"
    output.write_text(
        __import__("json").dumps(
            {
                "model_id": args.model_id,
                "task": "BANKING77",
                "device": str(device),
                "config": {
                    "seed": config.seed,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "max_length": config.max_length,
                },
                "train_samples": len(train_frame),
                "validation_samples": len(validation_frame),
                "best_epoch": selection.epoch,
                "best_validation_loss": selection.best_loss,
                "checkpoint_metadata": metadata,
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
