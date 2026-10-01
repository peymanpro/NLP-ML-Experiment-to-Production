import argparse
from pathlib import Path

from nlp_ml_lab.config import ProjectPaths
from nlp_ml_lab.data.loaders import load_banking77
from nlp_ml_lab.data.splits import split_frame
from nlp_ml_lab.data.validation import TextClassificationSchema, validate_text_classification_frame
from nlp_ml_lab.evaluation.classification import classification_metrics
from nlp_ml_lab.evaluation.reporting import metrics_as_dict
from nlp_ml_lab.models.classical import train_logistic_regression
from nlp_ml_lab.models.tree import train_random_forest_baseline
from nlp_ml_lab.reproducibility import seed_everything


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical BANKING77 baselines.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-train", type=int, default=2500)
    parser.add_argument("--max-validation", type=int, default=500)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    return parser.parse_args()


def _limit(frame, limit: int):
    if limit <= 0:
        raise ValueError("limits must be positive")
    return frame.head(min(limit, len(frame))).copy()


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)

    paths = ProjectPaths.from_root(Path.cwd())
    paths.create_runtime_directories()

    splits = load_banking77(cache_dir=args.cache_dir)
    train = split_frame(
        splits["train"],
        label_column="label",
        seed=args.seed,
        test_size=0.1,
        validation_size=0.1,
    )

    schema = TextClassificationSchema()
    validate_text_classification_frame(train.train, schema=schema)
    validate_text_classification_frame(train.validation, schema=schema)
    validate_text_classification_frame(splits["test"], schema=schema)

    train_frame = _limit(train.train, args.max_train)
    validation_frame = _limit(train.validation, args.max_validation)

    logistic = train_logistic_regression(
        train_frame["text"].tolist(),
        train_frame["label"].tolist(),
        random_state=args.seed,
    )
    logistic_pred = logistic.predict(validation_frame["text"].tolist())

    forest = train_random_forest_baseline(
        train_frame["text"].tolist(),
        train_frame["label"].tolist(),
        random_state=args.seed,
    )
    forest_pred = forest.predict(validation_frame["text"].tolist())

    results = {
        "logistic_regression": metrics_as_dict(
            classification_metrics(validation_frame["label"].tolist(), logistic_pred)
        ),
        "random_forest": metrics_as_dict(
            classification_metrics(validation_frame["label"].tolist(), forest_pred)
        ),
        "seed": args.seed,
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame),
    }

    output = paths.reports / "classical-baselines.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        __import__("json").dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(__import__("json").dumps(results, indent=2))


if __name__ == "__main__":
    main()
