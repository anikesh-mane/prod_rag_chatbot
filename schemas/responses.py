"""API response models."""

from datetime import datetime
from uuid import UUID

from pydantic import Field

from schemas.base import BaseSchema


class SourceDocument(BaseSchema):
    """Retrieved source document."""

    document_id: str
    content: str
    score: float = Field(..., description="Relevance score")
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ChatResponse(BaseSchema):
    """Response model for chat endpoint."""

    query_id: UUID
    answer: str
    language: str = Field(..., description="Detected/used language (ISO 639-1)")
    sources: list[SourceDocument] = Field(default_factory=list)
    tokens_used: int = Field(..., description="Total tokens consumed")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    model: str = Field(..., description="LLM model used for generation")
    conversation_id: str | None = None


class FeedbackResponse(BaseSchema):
    """Response model for feedback submission."""

    feedback_id: UUID
    status: str = "accepted"
    message: str = "Thank you for your feedback"


class TokenResponse(BaseSchema):
    """Response model for authentication tokens."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiry in seconds")


class IngestResponse(BaseSchema):
    """Response model for document ingestion."""

    document_id: UUID
    chunks_created: int
    status: str = "ingested"


class FileIngestResponse(BaseSchema):
    """Response model for file ingestion."""

    document_id: UUID
    filename: str
    file_type: str
    chunks_created: int
    status: str = "ingested"


class MetricsResponse(BaseSchema):
    """Response model for metrics endpoint."""

    total_queries: int
    avg_latency_ms: float
    avg_tokens_per_query: float
    cache_hit_rate: float
    error_rate: float
    period_start: datetime
    period_end: datetime


class UserResponse(BaseSchema):
    """Response model for user data."""

    user_id: UUID
    email: str
    role: str
    created_at: datetime
    last_login: datetime | None = None


# =============================================================================
# Admin Response Models
# =============================================================================


class ChunkSummary(BaseSchema):
    """Summary of a document chunk."""

    chunk_id: str
    content_preview: str = Field(..., description="First 200 characters of chunk content")
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class DocumentSummary(BaseSchema):
    """Summary of a document in the vector store."""

    document_id: str
    chunk_count: int
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class DocumentListResponse(BaseSchema):
    """Response for listing documents."""

    documents: list[DocumentSummary]
    total: int = Field(..., description="Total number of documents")
    limit: int
    offset: int


class DocumentDetailResponse(BaseSchema):
    """Detailed response for a single document."""

    document_id: str
    chunk_count: int
    chunks: list[ChunkSummary]
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class DeleteDocumentResponse(BaseSchema):
    """Response for document deletion."""

    document_id: str
    chunks_deleted: int
    status: str = "deleted"


class ReindexResponse(BaseSchema):
    """Response for document re-indexing."""

    document_id: str
    chunks_reindexed: int
    status: str = "reindexed"


class VectorStoreStatsResponse(BaseSchema):
    """Response for vector store statistics."""

    collection_name: str
    total_chunks: int
    dimension: int
    index_type: str
    metric_type: str
