from unittest.mock import Mock

import torch

from nlp_ml_lab.models.transformer import TransformerClassifier


def test_transformer_wrapper_returns_logits() -> None:
    underlying = Mock()
    underlying.return_value.logits = torch.randn(2, 4)

    wrapper = TransformerClassifier(underlying)
    logits = wrapper(
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[1, 1], [1, 1]]),
    )

    assert logits.shape == (2, 4)
    underlying.assert_called_once()
