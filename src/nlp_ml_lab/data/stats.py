from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetQualityStats:
    rows: int
    columns: int
    missing_text: int
    missing_label: int
    duplicate_rows: int
    label_count: int
    min_text_length: int
    max_text_length: int


def summarize_text_classification_frame(
    frame: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "label",
) -> DatasetQualityStats:
    if text_column not in frame.columns or label_column not in frame.columns:
        raise ValueError("frame does not satisfy text-classification schema")

    lengths = frame[text_column].fillna("").astype(str).str.len()

    return DatasetQualityStats(
        rows=len(frame),
        columns=len(frame.columns),
        missing_text=int(frame[text_column].isna().sum()),
        missing_label=int(frame[label_column].isna().sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        label_count=int(frame[label_column].nunique(dropna=True)),
        min_text_length=int(lengths.min()) if len(lengths) else 0,
        max_text_length=int(lengths.max()) if len(lengths) else 0,
    )
