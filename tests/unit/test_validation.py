import pandas as pd
import pytest

from nlp_ml_lab.data.validation import validate_text_classification_frame


def test_valid_classification_frame_passes() -> None:
    frame = pd.DataFrame(
        {
            "text": ["hello", "goodbye", "thanks"],
            "label": [0, 1, 0],
        }
    )

    validate_text_classification_frame(frame)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"text": ["hello"]}),
        pd.DataFrame({"label": [0, 1]}),
        pd.DataFrame({"text": ["hello", ""], "label": [0, 1]}),
        pd.DataFrame({"text": ["hello", None], "label": [0, 1]}),
        pd.DataFrame({"text": ["hello", "world"], "label": [0, None]}),
        pd.DataFrame({"text": ["hello", "world"], "label": [0, 0]}),
    ],
)
def test_invalid_classification_frame_is_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        validate_text_classification_frame(frame)
