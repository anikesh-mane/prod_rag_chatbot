"""API request models."""

from pydantic import Field

from schemas.base import BaseSchema


class ChatRequest(BaseSchema):
    """Request model for chat endpoint."""

    query: str = Field(..., min_length=1, max_length=4096, description="User's question")
    conversation_id: str | None = Field(
        default=None, description="Optional conversation ID for context"
    )
    language: str | None = Field(
        default=None,
        description="Preferred response language (ISO 639-1 code). Auto-detected if not provided.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source documents in response")


class FeedbackRequest(BaseSchema):
    """Request model for feedback submission."""

    query_id: str = Field(..., description="ID of the query being rated")
    rating: int = Field(..., ge=1, le=5, description="User rating (1-5)")
    comment: str | None = Field(default=None, max_length=1000, description="Optional comment")
    feedback_type: str = Field(
        default="rating", description="Type of feedback: rating, correction, flag"
    )


class DocumentIngestRequest(BaseSchema):
    """Request model for document ingestion."""

    source: str = Field(..., description="Document source identifier")
    content: str = Field(..., min_length=1, description="Document content")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Document metadata"
    )
    chunk_size: int = Field(default=512, ge=100, le=2048, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Overlap between chunks")


class TokenRefreshRequest(BaseSchema):
    """Request model for token refresh."""

    refresh_token: str = Field(..., description="Refresh token")


class DocumentListQuery(BaseSchema):
    """Query parameters for listing documents."""

    limit: int = Field(default=50, ge=1, le=500, description="Maximum documents to return")
    offset: int = Field(default=0, ge=0, description="Number of documents to skip")
