"""Authentication endpoints for user registration, login, and token refresh."""

from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from auth.jwt_handler import (
    JWTError,
    create_access_token,
    create_refresh_token,
    decode_access_token,
)
from auth.password import hash_password, verify_password
from configs import get_settings
from core.database import get_db_session
from memory.models import User
from schemas import (
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    LoginRequest,
    RegisterRequest,
    TokenRefreshRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {"model": ErrorResponse, "description": "Username or email already exists"},
    },
)
async def register(body: RegisterRequest) -> UserResponse:
    """Register a new user.

    Creates a new user account with a hashed password.
    Returns the created user profile (without password).
    """
    async with get_db_session() as session:
        # Check if username already exists
        existing = await session.execute(
            select(User).where(
                (User.username == body.username) | (User.email == body.email)
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=ErrorDetail(
                    code=ErrorCode.VALIDATION_ERROR,
                    message="Username or email already registered",
                ).model_dump(),
            )

        # Create user
        user = User(
            username=body.username,
            email=body.email,
            password_hash=hash_password(body.password),
        )
        session.add(user)
        await session.flush()

        logger.info("User registered", user_id=str(user.id), username=user.username)

        return UserResponse(
            user_id=user.id,
            email=user.email,
            role=user.role.value,
            created_at=user.created_at,
            last_login=None,
        )


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
    },
)
async def login(body: LoginRequest) -> TokenResponse:
    """Authenticate a user and return JWT tokens.

    Validates username and password against the database.
    Returns access and refresh tokens on success.
    """
    settings = get_settings()

    async with get_db_session() as session:
        # Look up user by username
        result = await session.execute(
            select(User).where(User.username == body.username)
        )
        user = result.scalar_one_or_none()

        if user is None or not verify_password(body.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ErrorDetail(
                    code=ErrorCode.AUTH_INVALID_TOKEN,
                    message="Invalid username or password",
                ).model_dump(),
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ErrorDetail(
                    code=ErrorCode.AUTH_INVALID_TOKEN,
                    message="Account is deactivated",
                ).model_dump(),
            )

        # Update last login timestamp
        user.last_login = datetime.utcnow()

        # Issue tokens
        access_token = create_access_token(
            user_id=str(user.id),
            email=user.email,
            role=user.role.value,
        )
        refresh_token = create_refresh_token(user_id=str(user.id))

        logger.info("User logged in", user_id=str(user.id), username=user.username)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.auth.access_token_expire_minutes * 60,
        )


@router.post(
    "/auth/refresh",
    response_model=TokenResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid refresh token"},
    },
)
async def refresh_token(body: TokenRefreshRequest) -> TokenResponse:
    """Refresh an access token using a valid refresh token.

    Looks up the user in the database to get current email/role,
    then issues a new access token.
    """
    settings = get_settings()

    try:
        payload = decode_access_token(body.refresh_token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorDetail(
                code=ErrorCode.AUTH_INVALID_TOKEN,
                message="Invalid or expired refresh token",
            ).model_dump(),
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorDetail(
                code=ErrorCode.AUTH_INVALID_TOKEN,
                message="Token is not a refresh token",
            ).model_dump(),
        )

    user_id = payload["sub"]

    # Look up current user state from DB
    async with get_db_session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ErrorDetail(
                    code=ErrorCode.AUTH_INVALID_TOKEN,
                    message="User not found or deactivated",
                ).model_dump(),
            )

        access_token = create_access_token(
            user_id=str(user.id),
            email=user.email,
            role=user.role.value,
        )

    logger.info("Token refreshed", user_id=user_id)

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.auth.access_token_expire_minutes * 60,
    )
