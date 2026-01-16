"""Health check endpoints."""

import time

import structlog
from fastapi import APIRouter

from schemas import ComponentHealth, HealthResponse, HealthStatus

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe - returns 200 if the process is running."""
    return HealthResponse(status=HealthStatus.HEALTHY)


@router.get("/ready", response_model=HealthResponse)
async def readiness_check() -> HealthResponse:
    """Readiness probe - checks connectivity to all dependencies."""
    components: list[ComponentHealth] = []
    overall_status = HealthStatus.HEALTHY

    # Check database connectivity
    db_health = await _check_database()
    components.append(db_health)
    if db_health.status != HealthStatus.HEALTHY:
        overall_status = HealthStatus.DEGRADED

    # Check Redis connectivity
    redis_health = await _check_redis()
    components.append(redis_health)
    if redis_health.status != HealthStatus.HEALTHY:
        overall_status = HealthStatus.DEGRADED

    # Check Milvus connectivity
    milvus_health = await _check_milvus()
    components.append(milvus_health)
    if milvus_health.status != HealthStatus.HEALTHY:
        overall_status = HealthStatus.DEGRADED

    # If any critical component is unhealthy, mark overall as unhealthy
    if any(c.status == HealthStatus.UNHEALTHY for c in components):
        overall_status = HealthStatus.UNHEALTHY

    return HealthResponse(status=overall_status, components=components)


async def _check_database() -> ComponentHealth:
    """Check PostgreSQL database connectivity."""
    start = time.perf_counter()
    try:
        # TODO: Implement actual database ping
        # async with get_db_session() as session:
        #     await session.execute(text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="postgresql",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
        )
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return ComponentHealth(
            name="postgresql",
            status=HealthStatus.UNHEALTHY,
            error=str(e),
        )


async def _check_redis() -> ComponentHealth:
    """Check Redis connectivity."""
    start = time.perf_counter()
    try:
        # TODO: Implement actual Redis ping
        # redis_client = get_redis_client()
        # await redis_client.ping()
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
        )
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        return ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            error=str(e),
        )


async def _check_milvus() -> ComponentHealth:
    """Check Milvus vector database connectivity."""
    start = time.perf_counter()
    try:
        # TODO: Implement actual Milvus health check
        # milvus_client = get_milvus_client()
        # await milvus_client.has_collection(settings.milvus.collection_name)
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="milvus",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
        )
    except Exception as e:
        logger.error("Milvus health check failed", error=str(e))
        return ComponentHealth(
            name="milvus",
            status=HealthStatus.UNHEALTHY,
            error=str(e),
        )
