"""Database migration script - creates conversation memory tables."""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from sqlalchemy import text

from configs import get_settings
from core.database import init_database, close_database, get_session_factory
from memory.models import Base

logger = structlog.get_logger(__name__)


async def create_tables() -> None:
    """Create all database tables defined in memory.models."""
    settings = get_settings()

    logger.info("Initializing database connection...")
    await init_database()

    factory = get_session_factory()
    engine = factory.kw["bind"]

    logger.info("Creating tables...")

    async with engine.begin() as conn:
        # Create all tables from Base metadata
        await conn.run_sync(Base.metadata.create_all)

    # Verify tables were created
    async with factory() as session:
        result = await session.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('conversations', 'messages')
                """
            )
        )
        tables = [row[0] for row in result.fetchall()]

    if "conversations" in tables and "messages" in tables:
        logger.info("Tables created successfully", tables=tables)
    else:
        logger.error("Table creation failed", found_tables=tables)
        raise RuntimeError("Failed to create required tables")

    await close_database()


async def drop_tables() -> None:
    """Drop all conversation memory tables (destructive!)."""
    logger.warning("Dropping all conversation memory tables...")

    await init_database()
    factory = get_session_factory()
    engine = factory.kw["bind"]

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    logger.info("Tables dropped")
    await close_database()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database migration for conversation memory")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing tables before creating (destructive!)",
    )
    args = parser.parse_args()

    async def main():
        if args.drop:
            await drop_tables()
        await create_tables()

    asyncio.run(main())
