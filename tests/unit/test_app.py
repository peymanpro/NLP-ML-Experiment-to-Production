from fastapi.testclient import TestClient

from nlp_ml_lab.serving.app import app

client = TestClient(app)


def test_health_endpoint_is_available() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint_reports_unloaded_model() -> None:
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_predict_requires_loaded_model() -> None:
    response = client.post("/predict", json={"text": "hello"})

    assert response.status_code == 503
