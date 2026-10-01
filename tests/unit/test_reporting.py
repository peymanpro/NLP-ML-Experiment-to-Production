from pathlib import Path

from nlp_ml_lab.evaluation.classification import ClassificationMetrics
from nlp_ml_lab.evaluation.reporting import write_metrics_json


def test_metrics_are_written_as_json(tmp_path: Path) -> None:
    target = tmp_path / "reports" / "metrics.json"
    metrics = ClassificationMetrics(
        accuracy=0.5,
        precision_macro=0.5,
        recall_macro=0.5,
        f1_macro=0.5,
    )

    write_metrics_json(metrics, target)

    assert target.exists()
    assert '"f1_macro": 0.5' in target.read_text(encoding="utf-8")
