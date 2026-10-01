from pathlib import Path

from nlp_ml_lab.mlops.registry import ModelRecord, ModelRegistry


def test_model_registry_registers_and_promotes(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    record = ModelRecord(
        model_name="intent-classifier",
        version="2026.10.01",
        artifact_path="models/best-model.pt",
        task="BANKING77",
        dataset_fingerprint="abc",
        code_revision="def",
        metrics={"f1_macro": 0.5},
    )

    registry.register(record)
    promoted = registry.promote(
        model_name="intent-classifier",
        version="2026.10.01",
        stage="production",
    )

    assert promoted.stage == "production"
    assert registry.list()[0].model_name == "intent-classifier"
