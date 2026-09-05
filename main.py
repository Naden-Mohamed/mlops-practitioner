import time
import logging
from fastapi import FastAPI, HTTPException
from prodml.api.middleware import CorrelationIDMiddleware
from prodml.config.logging_conf import setup_logging
from prodml.api.schemas import PredictRequest, PredictionResponse
from contextlib import asynccontextmanager
from prodml.predict import DurationPredictor
from prodml.config.config import get_settings

settings = get_settings()
setup_logging()
logger = logging.getLogger("prodml.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    duration_predictor = DurationPredictor(
        model_bundle_path=settings.linear_regression_model_path
    )
    app.state.feature_pipeline, app.state.model = duration_predictor.load()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CorrelationIDMiddleware)


@app.get("/health")
def check_health():
    if app.state.model:
        return {"status": "Healthy"}
    else:
        return {"status": "model is not loaded"}


@app.post("/predict")
async def predict(data: PredictRequest) -> PredictionResponse:
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
        prediction = app.state.model.predict(features)
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
        model_version=app.state.model.name,
        correlation_id="",
        latency_ms=latency_ms,
    )
