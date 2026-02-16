"""Authentication endpoints for token issuance and refresh."""

import structlog
from fastapi import APIRouter, HTTPException, status

from auth.jwt_handler import (
    JWTError,
    create_access_token,
    create_refresh_token,
    decode_access_token,
)
from configs import get_settings
from schemas import (
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    TokenRefreshRequest,
    TokenRequest,
    TokenResponse,
    UserRole,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/auth/token",
    response_model=TokenResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def create_token(body: TokenRequest) -> TokenResponse:
    """Issue access and refresh tokens.

    Accepts user credentials and returns a JWT access token
    and a refresh token. In production, this should validate
    against a user store before issuing tokens.
    """
    settings = get_settings()

    # Validate role
    try:
        role = UserRole(body.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"Invalid role: '{body.role}'. Must be 'user' or 'admin'.",
            ).model_dump(),
        )

    access_token = create_access_token(
        user_id=body.user_id,
        email=body.email,
        role=role.value,
    )
    refresh_token = create_refresh_token(user_id=body.user_id)

    logger.info(
        "Token issued",
        user_id=body.user_id,
        role=role.value,
    )

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

    The refresh token must be unexpired and of type 'refresh'.
    A new access token is returned; the refresh token is not rotated.
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

    # Ensure this is a refresh token
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorDetail(
                code=ErrorCode.AUTH_INVALID_TOKEN,
                message="Token is not a refresh token",
            ).model_dump(),
        )

    # Look up user info for the new access token.
    # In a real system, this would query the user store to get current email/role.
    # For now, we require the original user_id from the refresh token's 'sub' claim.
    user_id = payload["sub"]

    # Re-issue access token with same claims.
    # Note: Without a user store, we cannot populate email/role from DB.
    # The refresh endpoint is functional but limited until a user model exists.
    access_token = create_access_token(
        user_id=user_id,
        email=payload.get("email", ""),
        role=payload.get("role", "user"),
    )

    logger.info("Token refreshed", user_id=user_id)

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.auth.access_token_expire_minutes * 60,
    )
