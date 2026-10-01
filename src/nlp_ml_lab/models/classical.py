from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from nlp_ml_lab.features.tfidf import TfidfFeatures, fit_tfidf, transform_tfidf


@dataclass(frozen=True)
class LogisticRegressionBaseline:
    features: TfidfFeatures
    model: LogisticRegression

    def predict(self, texts: list[str]) -> list[int]:
        matrix = transform_tfidf(texts, vectorizer=self.features.vectorizer)
        return self.model.predict(matrix).astype(int).tolist()


def train_logistic_regression(
    texts: list[str],
    labels: list[int],
    *,
    random_state: int,
    max_iter: int = 1000,
) -> LogisticRegressionBaseline:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not texts:
        raise ValueError("training data must not be empty")
    if len(set(labels)) < 2:
        raise ValueError("training requires at least two classes")

    features = fit_tfidf(texts)
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(features.matrix, np.asarray(labels))
    return LogisticRegressionBaseline(features=features, model=model)
