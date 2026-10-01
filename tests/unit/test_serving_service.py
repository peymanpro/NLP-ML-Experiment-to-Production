from unittest.mock import Mock

import pytest

from nlp_ml_lab.observability.metrics import InferenceMetrics
from nlp_ml_lab.serving.service import InferenceService


def test_inference_service_delegates_to_predictor() -> None:
    predictor = Mock()
    predictor.model_version = "v1"
    predictor.predict.return_value = ("card_arrival", 0.91)

    service = InferenceService(
        predictor=predictor,
        metrics=InferenceMetrics(),
    )

    assert service.ready
    assert service.predict("where is my card") == ("card_arrival", 0.91)
    assert service.metrics.requests_total == 1


def test_inference_service_rejects_blank_input() -> None:
    service = InferenceService()

    with pytest.raises(ValueError):
        service.predict(" ")
