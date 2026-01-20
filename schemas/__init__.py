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
    DocumentListQuery,
    FeedbackRequest,
    TokenRefreshRequest,
)
from schemas.responses import (
    ChatResponse,
    ChunkSummary,
    DeleteDocumentResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    FeedbackResponse,
    FileIngestResponse,
    IngestResponse,
    MetricsResponse,
    ReindexResponse,
    SourceDocument,
    TokenResponse,
    UserResponse,
    VectorStoreStatsResponse,
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
    "DocumentListQuery",
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
    # Admin Responses
    "ChunkSummary",
    "DocumentSummary",
    "DocumentListResponse",
    "DocumentDetailResponse",
    "DeleteDocumentResponse",
    "ReindexResponse",
    "VectorStoreStatsResponse",
]
