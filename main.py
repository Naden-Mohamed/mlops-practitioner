import logging
from fastapi import FastAPI, Request
from prodml.api.middleware import CorrelationIDMiddleware
from prodml.config.logging_conf import setup_logging
from contextlib import asynccontextmanager
from prodml.predict import DurationPredictor
from prodml.config.config import get_settings
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prodml.config.logging_conf import get_correlation_id
from prodml.api.routes import route

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
app.include_router(route)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [
        {
            "field": ".".join(str(p) for p in err["loc"] if p != "body"),
            "message": err["msg"],
        }
        for err in exc.errors()
    ]

    logger.warning(
        "Validation error",
        extra={"extra_info": {"errors": errors, "path": request.url.path}},
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": errors,
            "correlation_id": get_correlation_id(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled exception",
        extra={"extra_fields": {"path": request.url.path}},
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred. Please try again shortly.",
            "correlation_id": get_correlation_id(),
        },
    )


@app.get("/")
async def root():
    return {"message": "Welcome to the main API entrypoint!"}
