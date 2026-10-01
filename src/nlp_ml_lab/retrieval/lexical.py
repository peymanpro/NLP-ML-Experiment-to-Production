from collections import Counter
from dataclasses import dataclass
from math import log
import re


_TOKEN_PATTERN = re.compile(r"[\w-]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class LexicalResult:
    document_id: str
    score: float


class BM25Retriever:
    """Small dependency-free BM25 implementation for a transparent baseline."""

    def __init__(
        self,
        document_ids: list[str],
        documents: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(document_ids) != len(documents):
            raise ValueError("document_ids and documents must have the same length")
        if not documents:
            raise ValueError("documents must not be empty")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.document_ids = document_ids
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokenized = [_tokens(document) for document in documents]
        self._lengths = [len(tokens) for tokens in self._tokenized]
        self._avgdl = sum(self._lengths) / len(self._lengths)
        self._doc_frequency: Counter[str] = Counter()
        for tokens in self._tokenized:
            self._doc_frequency.update(set(tokens))

    def search(self, query: str, *, k: int = 10) -> list[LexicalResult]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if k <= 0:
            raise ValueError("k must be positive")

        query_terms = set(_tokens(query))
        scored: list[LexicalResult] = []

        for document_id, tokens, length in zip(
            self.document_ids,
            self._tokenized,
            self._lengths,
        ):
            term_counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                df = self._doc_frequency[term]
                idf = log(1.0 + (len(self.documents) - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self._avgdl
                )
                score += idf * (frequency * (self.k1 + 1)) / denominator

            scored.append(LexicalResult(document_id=document_id, score=score))

        return sorted(
            scored,
            key=lambda item: (-item.score, item.document_id),
        )[:k]
