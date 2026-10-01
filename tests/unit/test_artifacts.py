from pathlib import Path

from nlp_ml_lab.mlops.artifacts import ArtifactManifest, read_manifest, write_manifest


def test_artifact_manifest_round_trip(tmp_path: Path) -> None:
    manifest = ArtifactManifest(
        model_name="intent-classifier",
        model_version="v1",
        task="BANKING77",
        dataset_fingerprint="abc",
        code_revision="def",
        framework="pytorch",
        artifact_file="model.pt",
    )
    path = tmp_path / "manifest.json"

    write_manifest(manifest, path)

    assert read_manifest(path) == manifest
