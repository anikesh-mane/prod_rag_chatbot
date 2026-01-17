"""FastAPI dependencies for dependency injection."""

from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from configs import Settings, get_settings
from ingestion import IngestionPipeline
from llm import RAGGenerator
from retrieval import MilvusVectorStore, Retriever
from schemas import UserRole

logger = structlog.get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


# =============================================================================
# Settings Dependency
# =============================================================================


def get_settings_dep() -> Settings:
    """Get application settings."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


# =============================================================================
# Service Singletons
# =============================================================================

_vector_store: MilvusVectorStore | None = None
_retriever: Retriever | None = None
_generator: RAGGenerator | None = None
_ingestion_pipeline: IngestionPipeline | None = None


async def get_vector_store() -> MilvusVectorStore:
    """Get or create vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = MilvusVectorStore()
        await _vector_store.connect()
    return _vector_store


async def get_retriever() -> Retriever:
    """Get or create retriever instance."""
    global _retriever
    if _retriever is None:
        vector_store = await get_vector_store()
        _retriever = Retriever(vector_store=vector_store)
        await _retriever.initialize()
    return _retriever


async def get_generator() -> RAGGenerator:
    """Get or create RAG generator instance."""
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator


async def get_ingestion_pipeline() -> IngestionPipeline:
    """Get or create ingestion pipeline instance."""
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline()
    return _ingestion_pipeline


# Type aliases for cleaner dependency injection
VectorStoreDep = Annotated[MilvusVectorStore, Depends(get_vector_store)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
GeneratorDep = Annotated[RAGGenerator, Depends(get_generator)]
IngestionDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]


# =============================================================================
# Authentication Dependencies
# =============================================================================


class CurrentUser:
    """Represents the current authenticated user."""

    def __init__(
        self,
        user_id: str,
        email: str,
        role: UserRole,
    ):
        self.user_id = user_id
        self.email = email
        self.role = role

    def has_role(self, role: UserRole) -> bool:
        """Check if user has the specified role."""
        if self.role == UserRole.ADMIN:
            return True  # Admin has all roles
        return self.role == role


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    settings: SettingsDep,
) -> CurrentUser:
    """Extract and validate current user from JWT token.

    Args:
        credentials: HTTP Bearer credentials.
        settings: Application settings.

    Returns:
        Current authenticated user.

    Raises:
        HTTPException: If authentication fails.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        # TODO: Implement actual JWT validation
        # For now, return a mock user for development
        if settings.is_development and token == "dev-token":
            return CurrentUser(
                user_id="dev-user-123",
                email="dev@example.com",
                role=UserRole.ADMIN,
            )

        # In production, decode and validate JWT
        # payload = jwt.decode(token, settings.auth.jwt_secret_key.get_secret_value(), algorithms=[settings.auth.jwt_algorithm])
        # return CurrentUser(
        #     user_id=payload["sub"],
        #     email=payload["email"],
        #     role=UserRole(payload["role"]),
        # )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except Exception as e:
        logger.warning("Authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    settings: SettingsDep,
) -> CurrentUser | None:
    """Get current user if authenticated, None otherwise."""
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials, settings)
    except HTTPException:
        return None


def require_role(required_role: UserRole):
    """Dependency factory to require a specific role."""

    async def role_checker(
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> CurrentUser:
        if not current_user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role.value}' required",
            )
        return current_user

    return role_checker


# Type aliases for auth dependencies
CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]
OptionalUserDep = Annotated[CurrentUser | None, Depends(get_current_user_optional)]
AdminUserDep = Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))]


# =============================================================================
# Cleanup
# =============================================================================


async def cleanup_services() -> None:
    """Cleanup service connections on shutdown."""
    global _vector_store, _retriever, _generator, _ingestion_pipeline

    if _retriever:
        await _retriever.close()
        _retriever = None

    if _vector_store:
        await _vector_store.disconnect()
        _vector_store = None

    _generator = None
    _ingestion_pipeline = None

    logger.info("Services cleaned up")
