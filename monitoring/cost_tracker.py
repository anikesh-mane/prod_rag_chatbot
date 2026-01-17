"""
LLM cost tracking and estimation.

Tracks token usage and calculates costs based on model pricing.
Pricing is configurable and updated periodically.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypedDict

import structlog

logger = structlog.get_logger(__name__)


class ModelPricing(TypedDict):
    """Pricing per 1K tokens for a model."""

    prompt: float  # Cost per 1K prompt tokens
    completion: float  # Cost per 1K completion tokens


# Default pricing (USD per 1K tokens)
# These should be loaded from config in production
DEFAULT_PRICING: dict[str, ModelPricing] = {
    # OpenAI models
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    # Embedding models
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
    "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.0},
    "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0},
    # Mistral models (self-hosted costs are infrastructure-based)
    "mistral-7b": {"prompt": 0.0, "completion": 0.0},
    "mistral-large": {"prompt": 0.008, "completion": 0.024},
    # Google Gemini models (pricing per 1K tokens)
    "gemini-2.0-flash": {"prompt": 0.0001, "completion": 0.0004},
    "gemini-2.0-flash-lite": {"prompt": 0.000075, "completion": 0.0003},
    "gemini-1.5-flash": {"prompt": 0.000075, "completion": 0.0003},
    "gemini-1.5-flash-8b": {"prompt": 0.0000375, "completion": 0.00015},
    "gemini-1.5-pro": {"prompt": 0.00125, "completion": 0.005},
    "gemini-1.0-pro": {"prompt": 0.0005, "completion": 0.0015},
    # Default fallback
    "default": {"prompt": 0.01, "completion": 0.03},
}


@dataclass
class UsageRecord:
    """Record of a single LLM API call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: datetime
    request_id: str | None = None
    user_id: str | None = None
    endpoint: str | None = None


class CostTracker:
    """Tracks LLM usage and costs.

    Maintains running totals and per-request records for cost analysis.
    """

    def __init__(self, pricing: dict[str, ModelPricing] | None = None) -> None:
        """Initialize the cost tracker.

        Args:
            pricing: Custom pricing table. Defaults to DEFAULT_PRICING.
        """
        self.pricing = pricing or DEFAULT_PRICING
        self._records: list[UsageRecord] = []
        self._totals: dict[str, dict[str, float]] = {}

    def get_pricing(self, model: str) -> ModelPricing:
        """Get pricing for a model, falling back to default if not found."""
        # Try exact match first
        if model in self.pricing:
            return self.pricing[model]

        # Try prefix matching (e.g., "gpt-4-0125-preview" -> "gpt-4")
        for key in self.pricing:
            if model.startswith(key):
                return self.pricing[key]

        logger.warning("Unknown model pricing, using default", model=model)
        return self.pricing.get("default", {"prompt": 0.01, "completion": 0.03})

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost for a single API call.

        Args:
            model: Model name.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.

        Returns:
            Cost in USD.
        """
        pricing = self.get_pricing(model)
        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]
        return prompt_cost + completion_cost

    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        request_id: str | None = None,
        user_id: str | None = None,
        endpoint: str | None = None,
    ) -> UsageRecord:
        """Record usage from an LLM API call.

        Args:
            model: Model name.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            request_id: Optional request ID for tracing.
            user_id: Optional user ID for attribution.
            endpoint: Optional endpoint name.

        Returns:
            The created usage record.
        """
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)

        record = UsageRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            timestamp=datetime.now(timezone.utc),
            request_id=request_id,
            user_id=user_id,
            endpoint=endpoint,
        )

        self._records.append(record)
        self._update_totals(record)

        logger.info(
            "LLM usage recorded",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 6),
            request_id=request_id,
        )

        return record

    def _update_totals(self, record: UsageRecord) -> None:
        """Update running totals with a new record."""
        if record.model not in self._totals:
            self._totals[record.model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "request_count": 0,
            }

        self._totals[record.model]["prompt_tokens"] += record.prompt_tokens
        self._totals[record.model]["completion_tokens"] += record.completion_tokens
        self._totals[record.model]["total_tokens"] += (
            record.prompt_tokens + record.completion_tokens
        )
        self._totals[record.model]["cost_usd"] += record.cost_usd
        self._totals[record.model]["request_count"] += 1

    def get_totals(self) -> dict[str, dict[str, float]]:
        """Get running totals by model."""
        return self._totals.copy()

    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        return sum(t["cost_usd"] for t in self._totals.values())

    def get_total_tokens(self) -> int:
        """Get total tokens across all models."""
        return int(sum(t["total_tokens"] for t in self._totals.values()))

    def get_records(
        self,
        limit: int = 100,
        model: str | None = None,
        user_id: str | None = None,
    ) -> list[UsageRecord]:
        """Get recent usage records with optional filtering.

        Args:
            limit: Maximum records to return.
            model: Filter by model name.
            user_id: Filter by user ID.

        Returns:
            List of usage records, most recent first.
        """
        records = self._records.copy()

        if model:
            records = [r for r in records if r.model == model]
        if user_id:
            records = [r for r in records if r.user_id == user_id]

        return sorted(records, key=lambda r: r.timestamp, reverse=True)[:limit]

    def get_summary(self) -> dict:
        """Get a summary of all usage.

        Returns:
            Dictionary with usage summary.
        """
        return {
            "total_cost_usd": round(self.get_total_cost(), 4),
            "total_tokens": self.get_total_tokens(),
            "total_requests": sum(
                int(t["request_count"]) for t in self._totals.values()
            ),
            "by_model": {
                model: {
                    "cost_usd": round(totals["cost_usd"], 4),
                    "prompt_tokens": int(totals["prompt_tokens"]),
                    "completion_tokens": int(totals["completion_tokens"]),
                    "total_tokens": int(totals["total_tokens"]),
                    "request_count": int(totals["request_count"]),
                }
                for model, totals in self._totals.items()
            },
        }

    def reset(self) -> None:
        """Reset all tracking data."""
        self._records.clear()
        self._totals.clear()


# Global cost tracker instance
cost_tracker = CostTracker()


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
) -> float:
    """Estimate cost for an API call without recording it.

    Useful for pre-call cost estimation.

    Args:
        model: Model name.
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Expected completion tokens (estimate).

    Returns:
        Estimated cost in USD.
    """
    return cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
