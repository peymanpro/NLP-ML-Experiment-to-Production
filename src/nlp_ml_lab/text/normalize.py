import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Apply conservative normalization without removing meaningful tokens."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.strip().lower()
    return _WHITESPACE.sub(" ", normalized)
