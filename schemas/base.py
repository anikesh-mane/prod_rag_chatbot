"""Base Pydantic models shared across the application."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp tracking."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None


class RequestIdMixin(BaseModel):
    """Mixin for models that need request ID tracking."""

    request_id: UUID = Field(default_factory=uuid4)


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""

    # Authentication errors
    AUTH_INVALID_TOKEN = "AUTH_INVALID_TOKEN"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_INSUFFICIENT_PERMISSIONS"
    AUTH_TOKEN_EXPIRED = "AUTH_TOKEN_EXPIRED"

    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Validation
    VALIDATION_ERROR = "VALIDATION_ERROR"

    # Retrieval errors
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    RETRIEVAL_NO_RESULTS = "RETRIEVAL_NO_RESULTS"

    # LLM errors
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    LLM_CONTENT_FILTERED = "LLM_CONTENT_FILTERED"
    LLM_GENERATION_FAILED = "LLM_GENERATION_FAILED"

    # Generic
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_FOUND = "NOT_FOUND"


class ErrorDetail(BaseSchema):
    """Detailed error information for API responses."""

    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None
    request_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseSchema):
    """Standard error response wrapper."""

    error: ErrorDetail


class UserRole(str, Enum):
    """User roles for RBAC."""

    USER = "user"
    ADMIN = "admin"


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class ComponentHealth(BaseSchema):
    """Health status for a single component."""

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseSchema):
    """Health check response."""

    status: HealthStatus
    components: list[ComponentHealth] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
