"""Monitoring and observability modules."""

from monitoring.cost_tracker import CostTracker, cost_tracker, estimate_cost
from monitoring.logging_config import get_logger, setup_logging
from monitoring.metrics import (
    MetricsRegistry,
    metrics,
    record_cache_access,
    record_feedback,
    record_http_request,
    record_ingestion,
    record_llm_call,
    record_llm_error,
    record_retrieval,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    # Metrics
    "MetricsRegistry",
    "metrics",
    "record_http_request",
    "record_retrieval",
    "record_llm_call",
    "record_llm_error",
    "record_cache_access",
    "record_feedback",
    "record_ingestion",
    # Cost tracking
    "CostTracker",
    "cost_tracker",
    "estimate_cost",
]
