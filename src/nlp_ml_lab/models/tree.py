from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

from nlp_ml_lab.features.tfidf import TfidfFeatures, fit_tfidf, transform_tfidf


@dataclass(frozen=True)
class RandomForestBaseline:
    features: TfidfFeatures
    reducer: TruncatedSVD
    model: RandomForestClassifier

    def predict(self, texts: list[str]) -> list[int]:
        matrix = transform_tfidf(texts, vectorizer=self.features.vectorizer)
        reduced = self.reducer.transform(matrix)
        return self.model.predict(reduced).astype(int).tolist()


def train_random_forest_baseline(
    texts: list[str],
    labels: list[int],
    *,
    random_state: int,
    components: int = 64,
    estimators: int = 100,
) -> RandomForestBaseline:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not texts:
        raise ValueError("training data must not be empty")
    if len(set(labels)) < 2:
        raise ValueError("training requires at least two classes")
    if components <= 0:
        raise ValueError("components must be positive")
    if estimators <= 0:
        raise ValueError("estimators must be positive")

    features = fit_tfidf(texts)
    max_components = min(features.matrix.shape[0] - 1, features.matrix.shape[1] - 1)
    if max_components < 1:
        raise ValueError("training data does not have enough dimensions for SVD")
    components = min(components, max_components)

    reducer = TruncatedSVD(
        n_components=components,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(features.matrix)

    model = RandomForestClassifier(
        n_estimators=estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(reduced, np.asarray(labels))

    return RandomForestBaseline(
        features=features,
        reducer=reducer,
        model=model,
    )
