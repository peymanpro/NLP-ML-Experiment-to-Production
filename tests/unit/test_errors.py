from nlp_ml_lab.evaluation.errors import confusion_pairs, summarize_errors


def test_error_summary_counts_mistakes() -> None:
    result = summarize_errors([0, 1, 1], [0, 0, 1])

    assert result.total == 3
    assert result.errors == 1
    assert result.accuracy == 2 / 3


def test_confusion_pairs_are_counted() -> None:
    pairs = confusion_pairs([0, 1, 1], [0, 0, 1])

    assert pairs[(1, 0)] == 1
    assert pairs[(0, 0)] == 1
