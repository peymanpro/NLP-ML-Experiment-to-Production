from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 11
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-3
    embedding_dim: int = 64

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
