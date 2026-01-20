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
