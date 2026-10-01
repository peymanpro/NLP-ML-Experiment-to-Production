from nlp_ml_lab.features.tfidf import fit_tfidf, transform_tfidf


def test_tfidf_fit_and_transform_share_vocabulary() -> None:
    features = fit_tfidf(["card arrived", "cash withdrawal"])

    transformed = transform_tfidf(
        ["card payment"],
        vectorizer=features.vectorizer,
    )

    assert features.matrix.shape[0] == 2
    assert transformed.shape[0] == 1
    assert transformed.shape[1] == features.matrix.shape[1]
