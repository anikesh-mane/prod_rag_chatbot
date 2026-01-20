"""Pydantic schemas for the RAG chatbot."""

from schemas.base import (
    BaseSchema,
    ComponentHealth,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    RequestIdMixin,
    TimestampMixin,
    UserRole,
)
from schemas.internal import (
    Chunk,
    Document,
    EmbeddingVersion,
    FeedbackRecord,
    LLMRequest,
    LLMResponse,
    QueryContext,
    RetrievalResult,
)
from schemas.requests import (
    ChatRequest,
    DocumentIngestRequest,
    FeedbackRequest,
    TokenRefreshRequest,
)
from schemas.responses import (
    ChatResponse,
    FeedbackResponse,
    FileIngestResponse,
    IngestResponse,
    MetricsResponse,
    SourceDocument,
    TokenResponse,
    UserResponse,
)

__all__ = [
    # Base
    "BaseSchema",
    "TimestampMixin",
    "RequestIdMixin",
    "ErrorCode",
    "ErrorDetail",
    "ErrorResponse",
    "UserRole",
    "HealthStatus",
    "ComponentHealth",
    "HealthResponse",
    # Internal
    "Document",
    "Chunk",
    "RetrievalResult",
    "LLMRequest",
    "LLMResponse",
    "QueryContext",
    "FeedbackRecord",
    "EmbeddingVersion",
    # Requests
    "ChatRequest",
    "FeedbackRequest",
    "DocumentIngestRequest",
    "TokenRefreshRequest",
    # Responses
    "ChatResponse",
    "FeedbackResponse",
    "FileIngestResponse",
    "SourceDocument",
    "TokenResponse",
    "IngestResponse",
    "MetricsResponse",
    "UserResponse",
]
