from pathlib import Path

from nlp_ml_lab.data.splits import split_frame
from nlp_ml_lab.evaluation.classification import classification_metrics
from nlp_ml_lab.models.classical import train_logistic_regression
from nlp_ml_lab.reproducibility import seed_everything


def run_demo() -> None:
    seed_everything(11)

    frame = __import__("pandas").DataFrame(
        {
            "text": [
                "card arrived today",
                "card is late",
                "cash withdrawal failed",
                "atm cash issue",
                "cash withdrawal problem",
                "my new card arrived",
                "cash machine failed",
                "my card has not arrived",
                "atm did not give cash",
                "card delivery is late",
                "cash withdrawal was rejected",
                "card delivery arrived",
            ],
            "label": [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    splits = split_frame(frame, label_column="label", seed=11)
    model = train_logistic_regression(
        splits.train["text"].tolist(),
        splits.train["label"].tolist(),
        random_state=11,
    )
    predictions = model.predict(splits.validation["text"].tolist())
    metrics = classification_metrics(
        splits.validation["label"].tolist(),
        predictions,
    )

    print(metrics)


if __name__ == "__main__":
    run_demo()
