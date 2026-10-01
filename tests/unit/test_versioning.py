import pytest

from nlp_ml_lab.mlops.versioning import validate_model_version


@pytest.mark.parametrize("version", ["1.0.0", "v1.2.3", "1.2.3-rc1"])
def test_valid_model_versions(version: str) -> None:
    assert validate_model_version(version) == version


def test_invalid_model_version_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_model_version("latest")
