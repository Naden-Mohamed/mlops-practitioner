import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from ..config.logging_conf import set_correlation_id


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract existing header or generate a new UUID4
        corr_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in contextvar for loggers to access
        set_correlation_id(cid=corr_id)

        response = await call_next(request)
        # Return header back to caller
        response.headers["X-Request-ID"] = corr_id

        return response
