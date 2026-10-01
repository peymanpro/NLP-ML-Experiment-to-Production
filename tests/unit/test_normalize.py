import pytest

from nlp_ml_lab.text.normalize import normalize_text


def test_normalize_text_is_conservative() -> None:
    assert normalize_text("  Hello\tWORLD  ") == "hello world"
    assert normalize_text("card-payment") == "card-payment"


def test_normalize_text_requires_string() -> None:
    with pytest.raises(TypeError):
        normalize_text(123)  # type: ignore[arg-type]
