import time
import logging
from fastapi import APIRouter, Request, HTTPException
from prodml.api.schemas import PredictRequest, PredictionResponse
from prodml.config.logging_conf import get_correlation_id
from prodml.config.logging_conf import setup_logging

route = APIRouter()
setup_logging()
logger = logging.getLogger("prodml.api")


@route.get("health")
def check_health(request: Request):
    if request.app.state.model:
        return {"status": "Healthy"}
    else:
        return {"status": "model is not loaded"}


@route.post("/predict")
async def predict(request: Request, data: PredictRequest) -> PredictionResponse:
    start_time = time.perf_counter()

    if data.trip_distance > 100:
        logger.warning(
            "Input value outside expected training bounds",
            extra={"extra_fields": {"trip_distance": data.trip_distance}},
        )

    features = [data.trip_distance, data.passenger_count]
    logger.debug(
        "Extracted feature vector for prediction",
        extra={"extra_fields": {"features": features}},
    )

    try:
        prediction = request.app.state.model.predict(features)
    except Exception:
        logger.exception("Model prediction failure")
        raise HTTPException(status_code=500, detail="Prediction failed")

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Prediction served successfully",
        extra={
            "extra_fields": {
                "prediction": prediction,
                "latency_ms": round(latency_ms, 2),
            }
        },
    )

    return PredictionResponse(
        prediction=prediction,
        model_version=request.app.state.model.name,
        correlation_id=get_correlation_id(),
        latency_ms=latency_ms,
    )
