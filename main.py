"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import cleanup_services
from api.middleware import RateLimitMiddleware, RequestContextMiddleware
from api.routes import chat, feedback, health, metrics
from configs import get_settings
from monitoring.logging_config import setup_logging
from retrieval.cache import RedisCache, close_cache, get_cache

settings = get_settings()
logger = structlog.get_logger(__name__)

# Global Redis cache for rate limiting (set during startup)
_redis_cache: RedisCache | None = None


def get_redis_cache() -> RedisCache | None:
    """Get the global Redis cache instance."""
    return _redis_cache


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events."""
    global _redis_cache

    # Startup
    setup_logging(settings.log_level)
    logger.info(
        "Starting application",
        environment=settings.environment,
        debug=settings.debug,
        api_version=settings.api_version,
    )

    # Initialize Redis cache
    try:
        cache = await get_cache()
        if cache.is_connected():
            _redis_cache = cache
            logger.info("Redis cache initialized")
    except Exception as e:
        logger.warning("Redis cache initialization failed", error=str(e))
        _redis_cache = None

    yield

    # Shutdown
    logger.info("Shutting down application")
    await cleanup_services()
    await close_cache()
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
    )

    # Add middleware (order matters - first added = outermost)
    # Request context middleware for request ID and timing
    app.add_middleware(RequestContextMiddleware)

    # Rate limiting middleware (uses Redis if available, lazy initialization)
    app.add_middleware(RateLimitMiddleware, cache_getter=get_redis_cache)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    # Health routes at root level (no prefix)
    app.include_router(health.router, tags=["Health"])

    # API routes with version prefix
    app.include_router(
        chat.router,
        prefix=settings.api_prefix,
        tags=["Chat"],
    )
    app.include_router(
        feedback.router,
        prefix=settings.api_prefix,
        tags=["Feedback"],
    )

    # Metrics routes at root level (for Prometheus scraping)
    app.include_router(metrics.router, tags=["Metrics"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
