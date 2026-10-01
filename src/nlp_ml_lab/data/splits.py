from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_frame(
    frame: pd.DataFrame,
    *,
    label_column: str,
    seed: int,
    test_size: float = 0.2,
    validation_size: float = 0.1,
) -> DatasetSplits:
    if frame.empty:
        raise ValueError("cannot split an empty dataframe")
    if label_column not in frame.columns:
        raise ValueError(f"missing label column: {label_column}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")

    train_validation, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame[label_column],
    )

    relative_validation_size = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        random_state=seed,
        stratify=train_validation[label_column],
    )

    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
