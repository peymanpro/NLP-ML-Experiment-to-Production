import numpy as np

from nlp_ml_lab.reproducibility import seed_everything


def test_seed_reproduces_numpy_values() -> None:
    seed_everything(123)
    first = np.random.random(8)

    seed_everything(123)
    second = np.random.random(8)

    np.testing.assert_allclose(first, second)
