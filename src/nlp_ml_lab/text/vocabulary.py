from collections import Counter
from dataclasses import dataclass

from nlp_ml_lab.text.normalize import normalize_text


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]
    unk_id: int
    pad_id: int

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        tokens = normalize_text(text).split()
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]


def build_vocabulary(
    texts: list[str],
    *,
    min_frequency: int = 1,
) -> Vocabulary:
    if not texts:
        raise ValueError("texts must not be empty")
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")

    counts = Counter(
        token
        for text in texts
        for token in normalize_text(text).split()
    )

    token_to_id = {"<pad>": 0, "<unk>": 1}
    for token in sorted(token for token, count in counts.items() if count >= min_frequency):
        token_to_id.setdefault(token, len(token_to_id))

    return Vocabulary(token_to_id=token_to_id, unk_id=1, pad_id=0)
