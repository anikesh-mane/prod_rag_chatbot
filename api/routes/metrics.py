"""Metrics endpoints for observability."""

from fastapi import APIRouter, Response

from api.dependencies import CurrentUserDep, SettingsDep
from monitoring.cost_tracker import cost_tracker
from monitoring.metrics import metrics
from schemas import UserRole

router = APIRouter()


@router.get("/metrics", response_class=Response)
async def get_prometheus_metrics(
    settings: SettingsDep,
) -> Response:
    """Expose metrics in Prometheus text format.

    This endpoint is typically scraped by Prometheus at regular intervals.
    In production, consider protecting this endpoint or using a separate port.
    """
    return Response(
        content=metrics.export_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/metrics/summary")
async def get_metrics_summary(
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> dict:
    """Get a JSON summary of key metrics.

    Requires admin role in production.
    """
    # In production, require admin role
    if not settings.is_development and current_user.role != UserRole.ADMIN:
        return {"error": "Admin access required"}

    cost_summary = cost_tracker.get_summary()

    return {
        "llm_usage": cost_summary,
        "note": "Full metrics available at /metrics endpoint in Prometheus format",
    }


@router.get("/metrics/costs")
async def get_cost_breakdown(
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> dict:
    """Get detailed LLM cost breakdown.

    Requires admin role in production.
    """
    if not settings.is_development and current_user.role != UserRole.ADMIN:
        return {"error": "Admin access required"}

    return cost_tracker.get_summary()
