import pandas as pd

from nlp_ml_lab.data.stats import summarize_text_classification_frame


def test_quality_stats_capture_basic_properties() -> None:
    frame = pd.DataFrame(
        {
            "text": ["short", "a much longer sentence", "short"],
            "label": [0, 1, 0],
        }
    )

    stats = summarize_text_classification_frame(frame)

    assert stats.rows == 3
    assert stats.columns == 2
    assert stats.duplicate_rows == 1
    assert stats.label_count == 2
    assert stats.min_text_length == 5
    assert stats.max_text_length == len("a much longer sentence")
