"""Rate limiting middleware using Redis."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog
from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from configs import get_settings

if TYPE_CHECKING:
    from retrieval.cache import RedisCache

logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits per user/endpoint.

    Uses RedisCache for rate limiting with sliding window algorithm.
    The cache is retrieved lazily via a getter function to support
    initialization after app startup.
    """

    def __init__(self, app, cache_getter=None):
        """Initialize rate limit middleware.

        Args:
            app: FastAPI application.
            cache_getter: Function that returns RedisCache or None.
        """
        super().__init__(app)
        self._cache_getter = cache_getter
        self._settings = get_settings()

    async def _get_cache(self) -> RedisCache | None:
        """Get cache instance lazily."""
        if self._cache_getter is None:
            return None
        try:
            cache = self._cache_getter()
            if cache and cache.is_connected():
                return cache
        except Exception:
            pass
        return None

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Get cache lazily
        cache = await self._get_cache()

        # Skip rate limiting if Redis not configured or not connected
        if cache is None:
            return await call_next(request)

        # Store cache for use in rate limit check
        self._cache = cache

        # Get user identifier (from auth or IP)
        user_id = self._get_user_identifier(request)
        endpoint = request.url.path

        # Check rate limit
        is_allowed, remaining, reset_time = await self._check_rate_limit(
            user_id, endpoint
        )

        if not is_allowed:
            logger.warning(
                "Rate limit exceeded",
                user_id=user_id,
                endpoint=endpoint,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(reset_time),
                    "X-RateLimit-Limit": str(self._get_limit_for_endpoint(endpoint)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                },
            )

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(
            self._get_limit_for_endpoint(endpoint)
        )
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response

    def _get_user_identifier(self, request: Request) -> str:
        """Get user identifier from request."""
        # Try to get from auth token first
        if hasattr(request.state, "user_id"):
            return request.state.user_id

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _get_limit_for_endpoint(self, endpoint: str) -> int:
        """Get rate limit for endpoint."""
        settings = self._settings.rate_limit

        if "/chat" in endpoint:
            return settings.chat_per_minute
        elif "/feedback" in endpoint:
            return settings.feedback_per_hour
        else:
            return settings.api_key_per_hour

    def _get_window_for_endpoint(self, endpoint: str) -> int:
        """Get rate limit window in seconds."""
        if "/chat" in endpoint:
            return 60  # 1 minute
        else:
            return 3600  # 1 hour

    async def _check_rate_limit(
        self, user_id: str, endpoint: str
    ) -> tuple[bool, int, int]:
        """Check if request is within rate limit.

        Uses the RedisCache.check_rate_limit method which implements
        sliding window rate limiting.

        Returns:
            Tuple of (is_allowed, remaining, reset_timestamp).
        """
        limit = self._get_limit_for_endpoint(endpoint)
        window = self._get_window_for_endpoint(endpoint)

        try:
            is_allowed, remaining = await self._cache.check_rate_limit(
                user_id=user_id,
                endpoint=endpoint,
                max_requests=limit,
                window_seconds=window,
            )
            reset_time = int(time.time()) + window
            return is_allowed, remaining, reset_time

        except Exception as e:
            # On Redis error, allow request but log warning
            logger.warning("Rate limit check failed", error=str(e))
            return True, limit, int(time.time()) + window
