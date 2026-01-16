"""FastAPI application entrypoint."""

import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health
from configs import get_settings
from monitoring.logging_config import setup_logging

settings = get_settings()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    setup_logging(settings.log_level)
    logger.info(
        "Starting application",
        environment=settings.environment,
        debug=settings.debug,
    )

    # TODO: Initialize connections
    # - Database connection pool
    # - Redis client
    # - Milvus client
    # - Load FAISS index if applicable

    yield

    # Shutdown
    logger.info("Shutting down application")
    # TODO: Close connections gracefully
    # - Close database connections
    # - Close Redis client
    # - Flush pending metrics


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])

    # TODO: Add more routers as they are implemented
    # app.include_router(chat.router, prefix=settings.api_prefix, tags=["Chat"])
    # app.include_router(feedback.router, prefix=settings.api_prefix, tags=["Feedback"])

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
