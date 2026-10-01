from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    *,
    path: Path,
    metadata: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    *,
    path: Path,
    map_location: str = "cpu",
) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")

    state = payload.get("model_state")
    metadata = payload.get("metadata")
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint model_state must be a mapping")
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        raise TypeError("checkpoint metadata must be a string dictionary")

    model.load_state_dict(state)
    return dict(metadata)
