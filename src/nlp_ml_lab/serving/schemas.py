from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4096)


class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str


class ServiceStatus(BaseModel):
    status: str
    model_version: str | None = None
