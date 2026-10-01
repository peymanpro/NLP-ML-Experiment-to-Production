from pathlib import Path

from nlp_ml_lab.evaluation.experiments import new_experiment, write_experiment


def test_experiment_record_contains_provenance(tmp_path: Path) -> None:
    record = new_experiment(
        run_id="run-001",
        model_id="baseline",
        task="classification",
        seed=11,
        metrics={"f1_macro": 0.5},
        parameters={"epochs": 3},
        dataset_fingerprint="abc",
        code_revision="def",
    )

    path = tmp_path / "experiment.json"
    write_experiment(record, path)

    assert path.exists()
    assert '"dataset_fingerprint": "abc"' in path.read_text(encoding="utf-8")
