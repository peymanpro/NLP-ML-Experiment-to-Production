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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pretrained Transformer BANKING77 baseline."
    )
    parser.add_argument("--model-id", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-train", type=int, default=2500)
    parser.add_argument("--max-validation", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
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
    seed_everything(args.seed)

    paths = ProjectPaths.from_root(Path.cwd())
    paths.create_runtime_directories()

    loaded = load_banking77(cache_dir=args.cache_dir)
    split = split_frame(
        loaded["train"],
        label_column="label",
        seed=args.seed,
        test_size=0.1,
        validation_size=0.1,
    )

    validation_frame = split.validation.head(args.max_validation).copy()

    bundle = load_sequence_classifier(
        args.model_id,
        num_labels=int(loaded["train"]["label"].nunique()),
    )
    validation_dataset = TransformerTextClassificationDataset(
        validation_frame["text"].tolist(),
        validation_frame["label"].tolist(),
        tokenizer=bundle.tokenizer,
        max_length=args.max_length,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(bundle.model).to(device)

    predictions = predict(model, validation_loader, device=device)
    metrics = classification_metrics(
        validation_frame["label"].tolist(),
        predictions,
    )

    output = paths.reports / "transformer-baseline.json"
    output.write_text(
        __import__("json").dumps(
            {
                "model_id": args.model_id,
                "device": str(device),
                "validation_samples": len(validation_frame),
                "max_length": args.max_length,
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
