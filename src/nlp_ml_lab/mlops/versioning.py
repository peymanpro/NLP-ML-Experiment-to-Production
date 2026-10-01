import re

_VERSION_PATTERN = re.compile(r"^v?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def validate_model_version(version: str) -> str:
    if not _VERSION_PATTERN.fullmatch(version):
        raise ValueError("model version must follow semantic-version form such as 1.2.3 or v1.2.3")
    return version
