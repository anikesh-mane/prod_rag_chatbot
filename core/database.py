"""Async database connection management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from configs import get_settings
from memory.models import Base

logger = structlog.get_logger(__name__)

# Global engine and session factory
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_database() -> None:
    """Initialize database engine and session factory."""
    global _engine, _session_factory

    if _engine is not None:
        return

    settings = get_settings()
    database_url = settings.database.url.get_secret_value()

    # Convert to async URL if needed (postgresql:// -> postgresql+asyncpg://)
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace(
            "postgresql://",
            "postgresql+asyncpg://",
            1,
        )

    _engine = create_async_engine(
        database_url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        echo=settings.database.echo,
        pool_pre_ping=True,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    # Create only conversations table
    async with _engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            checkfirst=True,
        )

    logger.info(
        "Database initialized",
        pool_size=settings.database.pool_size,
    )


async def close_database() -> None:
    """Close database connections."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the session factory.

    Returns:
        Session factory for creating database sessions.

    Raises:
        RuntimeError: If database not initialized.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session context manager.

    Usage:
        async with get_db_session() as session:
            # use session

    Yields:
        AsyncSession with automatic commit/rollback.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
