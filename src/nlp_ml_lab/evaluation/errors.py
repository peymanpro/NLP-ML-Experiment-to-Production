from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorSummary:
    total: int
    errors: int
    accuracy: float


def summarize_errors(y_true: list[int], y_pred: list[int]) -> ErrorSummary:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("error analysis requires at least one sample")

    errors = sum(actual != predicted for actual, predicted in zip(y_true, y_pred))
    total = len(y_true)
    return ErrorSummary(
        total=total,
        errors=errors,
        accuracy=(total - errors) / total,
    )


def confusion_pairs(y_true: list[int], y_pred: list[int]) -> dict[tuple[int, int], int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    counts: Counter[tuple[int, int]] = Counter()
    for actual, predicted in zip(y_true, y_pred):
        counts[(actual, predicted)] += 1
    return dict(counts)
