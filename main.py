import time
import logging
from fastapi import FastAPI, HTTPException
from prodml.api.middleware import CorellationIDMiddleware
from prodml.config.logging_conf import setup_logging
from prodml.api.schemas import PredictRequest
from contextlib import asynccontextmanager


setup_logging()
logger = logging.getLogger("prodml.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CorellationIDMiddleware)


@app.post("/health")
def check_health():
    return {"status": "Healthy"}


@app.post("/predict")
async def predict(data: PredictRequest) -> dict[str, float]:
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
        prediction = data.trip_distance * 2.5
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

    return {"prediction": prediction}
