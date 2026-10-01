import json
from dataclasses import asdict
from pathlib import Path

from nlp_ml_lab.evaluation.classification import ClassificationMetrics


def metrics_as_dict(metrics: ClassificationMetrics) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(metrics).items()}


def write_metrics_json(
    metrics: ClassificationMetrics,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics_as_dict(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
