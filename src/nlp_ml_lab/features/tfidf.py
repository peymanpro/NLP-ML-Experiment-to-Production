from dataclasses import dataclass

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_ml_lab.text.normalize import normalize_text


@dataclass(frozen=True)
class TfidfFeatures:
    matrix: csr_matrix
    vectorizer: TfidfVectorizer


def fit_tfidf(texts: list[str]) -> TfidfFeatures:
    if not texts:
        raise ValueError("texts must not be empty")

    vectorizer = TfidfVectorizer(
        lowercase=False,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform([normalize_text(text) for text in texts])
    return TfidfFeatures(matrix=matrix.tocsr(), vectorizer=vectorizer)


def transform_tfidf(
    texts: list[str],
    *,
    vectorizer: TfidfVectorizer,
) -> csr_matrix:
    if not texts:
        raise ValueError("texts must not be empty")
    return vectorizer.transform([normalize_text(text) for text in texts]).tocsr()
