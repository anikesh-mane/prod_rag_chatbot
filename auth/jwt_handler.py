"""JWT token creation and validation utilities."""

from datetime import datetime, timedelta, timezone

from jose import ExpiredSignatureError, JWTError, jwt

from configs import get_settings


def create_access_token(user_id: str, email: str, role: str) -> str:
    """Create a JWT access token.

    Args:
        user_id: Unique user identifier (stored as 'sub' claim).
        email: User email address.
        role: User role (e.g. 'user', 'admin').

    Returns:
        Encoded JWT access token string.
    """
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.auth.access_token_expire_minutes,
    )
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(
        payload,
        settings.auth.jwt_secret_key.get_secret_value(),
        algorithm=settings.auth.jwt_algorithm,
    )


def create_refresh_token(user_id: str) -> str:
    """Create a longer-lived JWT refresh token.

    Args:
        user_id: Unique user identifier (stored as 'sub' claim).

    Returns:
        Encoded JWT refresh token string.
    """
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.auth.refresh_token_expire_days,
    )
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh",
    }
    return jwt.encode(
        payload,
        settings.auth.jwt_secret_key.get_secret_value(),
        algorithm=settings.auth.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT token.

    Args:
        token: Encoded JWT token string.

    Returns:
        Decoded token payload as dict.

    Raises:
        ExpiredSignatureError: If the token has expired.
        JWTError: If the token is invalid or malformed.
    """
    settings = get_settings()
    return jwt.decode(
        token,
        settings.auth.jwt_secret_key.get_secret_value(),
        algorithms=[settings.auth.jwt_algorithm],
    )


__all__ = [
    "create_access_token",
    "create_refresh_token",
    "decode_access_token",
    "ExpiredSignatureError",
    "JWTError",
]
