from fastapi import FastAPI, HTTPException, Response, status

from nlp_ml_lab.observability.metrics import InferenceMetrics
from nlp_ml_lab.serving.schemas import PredictionRequest, PredictionResponse, ServiceStatus
from nlp_ml_lab.serving.service import InferenceService

service = InferenceService(metrics=InferenceMetrics())

app = FastAPI(
    title="NLP ML Experiment-to-Production",
    version="0.1.0",
)


@app.get("/health", response_model=ServiceStatus)
def health() -> ServiceStatus:
    return ServiceStatus(
        status="ok",
        model_version=service.model_version,
    )


@app.get("/ready", response_model=ServiceStatus)
def ready(response: Response) -> ServiceStatus:
    if not service.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ServiceStatus(
            status="not_ready",
            model_version=None,
        )

    return ServiceStatus(
        status="ready",
        model_version=service.model_version,
    )


@app.get("/metrics")
def metrics() -> dict[str, float | int]:
    telemetry = service.metrics
    if telemetry is None:
        return {"requests_total": 0, "errors_total": 0, "average_latency_ms": 0.0}
    return {
        "requests_total": telemetry.requests_total,
        "errors_total": telemetry.errors_total,
        "average_latency_ms": telemetry.average_latency_ms,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        label, confidence = service.predict(request.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(
        label=label,
        confidence=confidence,
        model_version=service.model_version or "unknown",
    )
