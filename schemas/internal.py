"""Internal data models used across modules."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import Field

from schemas.base import BaseSchema, TimestampMixin


class Document(BaseSchema, TimestampMixin):
    """Internal document representation."""

    document_id: UUID = Field(default_factory=uuid4)
    source: str
    content: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class Chunk(BaseSchema):
    """Text chunk with embedding reference."""

    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    index: int = Field(..., description="Position in original document")
    embedding_id: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RetrievalResult(BaseSchema):
    """Result from vector retrieval."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class LLMRequest(BaseSchema):
    """Internal request to LLM."""

    prompt: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    stop_sequences: list[str] = Field(default_factory=list)


class LLMResponse(BaseSchema):
    """Internal response from LLM."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    latency_ms: float


class QueryContext(BaseSchema):
    """Context for a user query through the pipeline."""

    query_id: UUID = Field(default_factory=uuid4)
    original_query: str
    processed_query: str | None = None
    detected_language: str | None = None
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    prompt: str | None = None
    response: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackRecord(BaseSchema, TimestampMixin):
    """Stored feedback record."""

    feedback_id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    user_id: UUID | None = None
    original_query: str
    retrieved_context: list[str] = Field(default_factory=list)
    generated_answer: str
    rating: int
    comment: str | None = None
    feedback_type: str = "rating"


class EmbeddingVersion(BaseSchema):
    """Embedding model version tracking."""

    version_id: str
    model_name: str
    dimension: int
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
