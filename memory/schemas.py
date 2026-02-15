"""Pydantic schemas for memory operations."""

from datetime import datetime
from uuid import UUID

from pydantic import Field

from schemas.base import BaseSchema


class MessageCreate(BaseSchema):
    """Schema for creating a new message."""

    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    query_id: UUID | None = None
    sources: list[dict] | None = None
    tokens_used: int | None = None
    model: str | None = None


class MessageResponse(BaseSchema):
    """Schema for message in API response."""

    id: UUID
    role: str
    content: str
    created_at: datetime
    query_id: UUID | None = None
    tokens_used: int | None = None
    model: str | None = None


class ConversationCreate(BaseSchema):
    """Schema for creating a new conversation."""

    user_id: str | None = None
    title: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ConversationResponse(BaseSchema):
    """Schema for conversation in API response."""

    id: UUID
    user_id: str | None
    title: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime
    is_active: bool


class ConversationHistoryResponse(BaseSchema):
    """Full conversation with messages."""

    conversation: ConversationResponse
    messages: list[MessageResponse]


class ConversationContext(BaseSchema):
    """Context for LLM generation (formatted history)."""

    conversation_id: UUID
    messages: list[dict[str, str]]  # [{"role": "user", "content": "..."}]
    message_count: int
