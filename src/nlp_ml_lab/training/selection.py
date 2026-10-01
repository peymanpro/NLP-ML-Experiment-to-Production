from dataclasses import dataclass
from math import inf
from pathlib import Path

from torch import nn

from nlp_ml_lab.training.checkpoints import save_checkpoint


@dataclass
class BestCheckpoint:
    best_loss: float = inf
    epoch: int = 0
    path: Path | None = None

    def consider(
        self,
        *,
        model: nn.Module,
        validation_loss: float,
        epoch: int,
        directory: Path,
        metadata: dict[str, str],
    ) -> bool:
        if validation_loss >= self.best_loss:
            return False

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "best-model.pt"
        save_checkpoint(
            model,
            path=path,
            metadata={
                **metadata,
                "epoch": str(epoch),
                "validation_loss": f"{validation_loss:.8f}",
            },
        )
        self.best_loss = validation_loss
        self.epoch = epoch
        self.path = path
        return True
