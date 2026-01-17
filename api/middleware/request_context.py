"""Request context middleware for request ID, timing, and metrics."""

import time
from uuid import uuid4

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from monitoring.metrics import record_http_request

logger = structlog.get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID and timing to all requests."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid4()))

        # Store in request state for access in routes
        request.state.request_id = request_id

        # Start timing
        start_time = time.perf_counter()

        # Bind request context to structlog
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        logger.info("Request started")

        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception and re-raise
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Request failed",
                duration_ms=round(duration_ms, 2),
                error=str(e),
            )
            raise

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        # Record metrics
        record_http_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration_ms / 1000,  # Convert to seconds for histogram
        )

        logger.info(
            "Request completed",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        return response
