import pandas as pd
import pytest

from nlp_ml_lab.data.splits import split_frame


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"sample-{index}" for index in range(30)],
            "label": [index % 3 for index in range(30)],
        }
    )


def test_split_frame_is_reproducible() -> None:
    frame = make_frame()

    first = split_frame(frame, label_column="label", seed=7)
    second = split_frame(frame, label_column="label", seed=7)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)


@pytest.mark.parametrize(
    ("test_size", "validation_size"),
    [
        (0.0, 0.1),
        (0.2, 0.0),
        (0.8, 0.3),
    ],
)
def test_split_frame_rejects_invalid_sizes(
    test_size: float, validation_size: float
) -> None:
    with pytest.raises(ValueError):
        split_frame(
            make_frame(),
            label_column="label",
            seed=7,
            test_size=test_size,
            validation_size=validation_size,
        )
