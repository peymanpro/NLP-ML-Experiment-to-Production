from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TextClassificationSchema:
    text_column: str = "text"
    label_column: str = "label"


def validate_text_classification_frame(
    frame: pd.DataFrame,
    *,
    schema: TextClassificationSchema = TextClassificationSchema(),
) -> None:
    """Validate the minimum contract required by classification components."""

    required = {schema.text_column, schema.label_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if frame.empty:
        raise ValueError("dataset must not be empty")

    text_values = frame[schema.text_column]
    if text_values.isna().any():
        raise ValueError("text column contains missing values")
    if (text_values.astype(str).str.strip() == "").any():
        raise ValueError("text column contains blank values")

    label_values = frame[schema.label_column]
    if label_values.isna().any():
        raise ValueError("label column contains missing values")
    if label_values.nunique(dropna=False) < 2:
        raise ValueError("classification dataset must contain at least two labels")
