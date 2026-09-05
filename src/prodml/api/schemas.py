from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    trip_distance: float = Field(..., gt=0, lt=200)
    passenger_count: int = Field(..., gt=0, lt=6)


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float
    batch_variants: int
