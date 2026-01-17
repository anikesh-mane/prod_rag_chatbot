"""Prometheus-style metrics for the RAG chatbot.

Tracks:
- Request latency (HTTP, retrieval, LLM)
- Token usage and costs
- Cache hit rates
- Error counts
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Callable, Generator

import structlog

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics we collect."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricValue:
    """Container for a single metric value."""

    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class HistogramValue:
    """Container for histogram metric data."""

    count: int = 0
    sum: float = 0.0
    buckets: dict[float, int] = field(default_factory=dict)

    def observe(self, value: float, bucket_boundaries: list[float]) -> None:
        """Record an observation in the histogram."""
        self.count += 1
        self.sum += value
        for boundary in bucket_boundaries:
            if value <= boundary:
                self.buckets[boundary] = self.buckets.get(boundary, 0) + 1


# Default histogram buckets for latency (in seconds)
DEFAULT_LATENCY_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

# Token count buckets
TOKEN_BUCKETS = [10, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000]


class MetricsRegistry:
    """Central registry for all application metrics.

    Thread-safe singleton that collects and exposes metrics
    in Prometheus text format.
    """

    _instance: MetricsRegistry | None = None
    _lock: Lock = Lock()

    def __new__(cls) -> MetricsRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the metrics registry."""
        self._counters: dict[str, dict[str, float]] = {}
        self._gauges: dict[str, dict[str, float]] = {}
        self._histograms: dict[str, dict[str, HistogramValue]] = {}
        self._metric_help: dict[str, str] = {}
        self._metric_type: dict[str, MetricType] = {}
        self._data_lock = Lock()

        # Register default metrics
        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        """Register standard metrics for the application."""
        # HTTP metrics
        self.register_histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds",
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.register_counter(
            "http_requests_total",
            "Total HTTP requests",
        )
        self.register_counter(
            "http_request_errors_total",
            "Total HTTP request errors",
        )

        # RAG pipeline metrics
        self.register_histogram(
            "retrieval_duration_seconds",
            "Vector retrieval latency in seconds",
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.register_histogram(
            "reranking_duration_seconds",
            "Reranking latency in seconds",
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.register_histogram(
            "llm_generation_duration_seconds",
            "LLM generation latency in seconds",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
        )
        self.register_counter(
            "retrieval_requests_total",
            "Total retrieval requests",
        )
        self.register_gauge(
            "retrieval_results_count",
            "Number of documents retrieved",
        )

        # Token and cost metrics
        self.register_counter(
            "llm_tokens_total",
            "Total LLM tokens used",
        )
        self.register_histogram(
            "llm_prompt_tokens",
            "Prompt token count distribution",
            buckets=TOKEN_BUCKETS,
        )
        self.register_histogram(
            "llm_completion_tokens",
            "Completion token count distribution",
            buckets=TOKEN_BUCKETS,
        )
        self.register_counter(
            "llm_cost_dollars_total",
            "Total LLM cost in dollars",
        )

        # Cache metrics
        self.register_counter(
            "cache_hits_total",
            "Total cache hits",
        )
        self.register_counter(
            "cache_misses_total",
            "Total cache misses",
        )

        # Ingestion metrics
        self.register_counter(
            "documents_ingested_total",
            "Total documents ingested",
        )
        self.register_counter(
            "chunks_created_total",
            "Total chunks created during ingestion",
        )
        self.register_histogram(
            "embedding_duration_seconds",
            "Embedding generation latency in seconds",
            buckets=DEFAULT_LATENCY_BUCKETS,
        )

        # Feedback metrics
        self.register_counter(
            "feedback_submissions_total",
            "Total feedback submissions",
        )
        self.register_gauge(
            "feedback_average_rating",
            "Rolling average feedback rating",
        )

        # Error metrics
        self.register_counter(
            "llm_errors_total",
            "Total LLM errors by type",
        )
        self.register_counter(
            "retrieval_errors_total",
            "Total retrieval errors",
        )

    def register_counter(self, name: str, help_text: str) -> None:
        """Register a counter metric."""
        with self._data_lock:
            self._metric_help[name] = help_text
            self._metric_type[name] = MetricType.COUNTER
            if name not in self._counters:
                self._counters[name] = {}

    def register_gauge(self, name: str, help_text: str) -> None:
        """Register a gauge metric."""
        with self._data_lock:
            self._metric_help[name] = help_text
            self._metric_type[name] = MetricType.GAUGE
            if name not in self._gauges:
                self._gauges[name] = {}

    def register_histogram(
        self,
        name: str,
        help_text: str,
        buckets: list[float] | None = None,
    ) -> None:
        """Register a histogram metric."""
        with self._data_lock:
            self._metric_help[name] = help_text
            self._metric_type[name] = MetricType.HISTOGRAM
            if name not in self._histograms:
                self._histograms[name] = {"_buckets": buckets or DEFAULT_LATENCY_BUCKETS}  # type: ignore

    def _labels_to_key(self, labels: dict[str, str] | None) -> str:
        """Convert labels dict to a sortable string key."""
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric."""
        key = self._labels_to_key(labels)
        with self._data_lock:
            if name not in self._counters:
                self._counters[name] = {}
            self._counters[name][key] = self._counters[name].get(key, 0.0) + value

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric value."""
        key = self._labels_to_key(labels)
        with self._data_lock:
            if name not in self._gauges:
                self._gauges[name] = {}
            self._gauges[name][key] = value

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record an observation in a histogram."""
        key = self._labels_to_key(labels)
        with self._data_lock:
            if name not in self._histograms:
                return
            buckets = self._histograms[name].get("_buckets", DEFAULT_LATENCY_BUCKETS)
            if key not in self._histograms[name]:
                self._histograms[name][key] = HistogramValue(
                    buckets={b: 0 for b in buckets}
                )
            hist = self._histograms[name][key]
            if isinstance(hist, HistogramValue):
                hist.observe(value, buckets)

    @contextmanager
    def timer(
        self,
        histogram_name: str,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager to time operations and record to histogram."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe_histogram(histogram_name, duration, labels)

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines: list[str] = []

        with self._data_lock:
            # Export counters
            for name, values in self._counters.items():
                if name in self._metric_help:
                    lines.append(f"# HELP {name} {self._metric_help[name]}")
                    lines.append(f"# TYPE {name} counter")
                for key, value in values.items():
                    label_str = f"{{{key}}}" if key else ""
                    lines.append(f"{name}{label_str} {value}")

            # Export gauges
            for name, values in self._gauges.items():
                if name in self._metric_help:
                    lines.append(f"# HELP {name} {self._metric_help[name]}")
                    lines.append(f"# TYPE {name} gauge")
                for key, value in values.items():
                    label_str = f"{{{key}}}" if key else ""
                    lines.append(f"{name}{label_str} {value}")

            # Export histograms
            for name, values in self._histograms.items():
                if name in self._metric_help:
                    lines.append(f"# HELP {name} {self._metric_help[name]}")
                    lines.append(f"# TYPE {name} histogram")
                buckets = values.get("_buckets", DEFAULT_LATENCY_BUCKETS)
                for key, hist in values.items():
                    if key == "_buckets":
                        continue
                    if not isinstance(hist, HistogramValue):
                        continue
                    label_prefix = f"{key}," if key else ""
                    for bucket in sorted(buckets):
                        bucket_count = hist.buckets.get(bucket, 0)
                        lines.append(
                            f'{name}_bucket{{{label_prefix}le="{bucket}"}} {bucket_count}'
                        )
                    lines.append(
                        f'{name}_bucket{{{label_prefix}le="+Inf"}} {hist.count}'
                    )
                    lines.append(f"{name}_sum{{{key}}} {hist.sum}" if key else f"{name}_sum {hist.sum}")
                    lines.append(f"{name}_count{{{key}}} {hist.count}" if key else f"{name}_count {hist.count}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._data_lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._register_default_metrics()


# Global metrics registry instance
metrics = MetricsRegistry()


# Convenience functions for common metric operations
def record_http_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
) -> None:
    """Record an HTTP request in metrics."""
    labels = {"method": method, "path": path, "status": str(status_code)}
    metrics.inc_counter("http_requests_total", labels=labels)
    metrics.observe_histogram("http_request_duration_seconds", duration, labels=labels)
    if status_code >= 400:
        metrics.inc_counter("http_request_errors_total", labels=labels)


def record_retrieval(
    duration: float,
    result_count: int,
    collection: str = "default",
) -> None:
    """Record a retrieval operation in metrics."""
    labels = {"collection": collection}
    metrics.inc_counter("retrieval_requests_total", labels=labels)
    metrics.observe_histogram("retrieval_duration_seconds", duration, labels=labels)
    metrics.set_gauge("retrieval_results_count", result_count, labels=labels)


def record_llm_call(
    duration: float,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    cost: float | None = None,
) -> None:
    """Record an LLM API call in metrics."""
    labels = {"model": model}
    metrics.observe_histogram("llm_generation_duration_seconds", duration, labels=labels)
    metrics.observe_histogram("llm_prompt_tokens", prompt_tokens, labels=labels)
    metrics.observe_histogram("llm_completion_tokens", completion_tokens, labels=labels)
    metrics.inc_counter(
        "llm_tokens_total",
        prompt_tokens + completion_tokens,
        labels={"model": model, "type": "total"},
    )
    metrics.inc_counter(
        "llm_tokens_total",
        prompt_tokens,
        labels={"model": model, "type": "prompt"},
    )
    metrics.inc_counter(
        "llm_tokens_total",
        completion_tokens,
        labels={"model": model, "type": "completion"},
    )
    if cost is not None:
        metrics.inc_counter("llm_cost_dollars_total", cost, labels=labels)


def record_llm_error(model: str, error_type: str) -> None:
    """Record an LLM error in metrics."""
    metrics.inc_counter("llm_errors_total", labels={"model": model, "error": error_type})


def record_cache_access(hit: bool, cache_type: str = "default") -> None:
    """Record a cache access (hit or miss)."""
    labels = {"cache": cache_type}
    if hit:
        metrics.inc_counter("cache_hits_total", labels=labels)
    else:
        metrics.inc_counter("cache_misses_total", labels=labels)


def record_feedback(rating: int, feedback_type: str) -> None:
    """Record a feedback submission."""
    metrics.inc_counter(
        "feedback_submissions_total",
        labels={"type": feedback_type, "rating": str(rating)},
    )


def record_ingestion(document_count: int, chunk_count: int, source_type: str) -> None:
    """Record document ingestion metrics."""
    labels = {"source": source_type}
    metrics.inc_counter("documents_ingested_total", document_count, labels=labels)
    metrics.inc_counter("chunks_created_total", chunk_count, labels=labels)
