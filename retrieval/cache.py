"""Redis caching layer for RAG components.

Implements caching strategies per architecture doc:
- LLM responses: 1 hour TTL, query hash key
- Embeddings: 24 hours TTL, text hash key
- Rate limiting: 1 minute TTL, sliding window
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any

import structlog

from configs import get_settings
from monitoring.metrics import record_cache_access

logger = structlog.get_logger(__name__)


def _hash_text(text: str) -> str:
    """Create a hash of text for cache keys."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class CacheConfig:
    """Cache TTL and key configuration."""

    # TTLs per architecture doc
    LLM_RESPONSE_TTL = timedelta(hours=1)
    EMBEDDING_TTL = timedelta(hours=24)
    SESSION_TTL = timedelta(minutes=15)
    RATE_LIMIT_TTL = timedelta(minutes=1)

    # Key prefixes following pattern: {service}:{entity_type}:{identifier}:{version}
    LLM_PREFIX = "rag:llm"
    EMBEDDING_PREFIX = "rag:emb"
    SESSION_PREFIX = "rag:session"
    RATE_LIMIT_PREFIX = "rag:ratelimit"


class RedisCache:
    """Async Redis cache client.

    Provides typed caching methods for LLM responses, embeddings,
    and session data with configurable TTLs.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        db: int = 0,
        ssl: bool = False,
    ):
        """Initialize Redis cache.

        Args:
            host: Redis host.
            port: Redis port.
            password: Redis password.
            db: Redis database number.
            ssl: Enable SSL/TLS.
        """
        self._host = host
        self._port = port
        self._password = password
        self._db = db
        self._ssl = ssl
        self._client: Any = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return

        try:
            import redis.asyncio as redis

            self._client = redis.Redis(
                host=self._host,
                port=self._port,
                password=self._password,
                db=self._db,
                ssl=self._ssl,
                decode_responses=True,
            )
            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info("Redis connected", host=self._host, port=self._port)

        except ImportError:
            logger.warning("redis package not installed, caching disabled")
        except Exception as e:
            logger.warning("Redis connection failed", error=str(e))
            self._connected = False

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client and self._connected:
            await self._client.close()
            self._connected = False
            logger.info("Redis disconnected")

    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected

    async def _get(self, key: str) -> str | None:
        """Get value from cache."""
        if not self._connected or not self._client:
            return None
        try:
            return await self._client.get(key)
        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return None

    async def _set(
        self,
        key: str,
        value: str,
        ttl: timedelta | None = None,
    ) -> bool:
        """Set value in cache."""
        if not self._connected or not self._client:
            return False
        try:
            if ttl:
                await self._client.setex(key, int(ttl.total_seconds()), value)
            else:
                await self._client.set(key, value)
            return True
        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False

    async def _delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._connected or not self._client:
            return False
        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning("Cache delete failed", key=key, error=str(e))
            return False

    # LLM Response Caching

    def _llm_cache_key(self, query_hash: str, model: str) -> str:
        """Generate LLM cache key."""
        return f"{CacheConfig.LLM_PREFIX}:{query_hash}:{model}"

    async def get_llm_response(
        self,
        query: str,
        model: str,
    ) -> str | None:
        """Get cached LLM response.

        Args:
            query: Original query text.
            model: Model name used.

        Returns:
            Cached response or None.
        """
        query_hash = _hash_text(query)
        key = self._llm_cache_key(query_hash, model)
        result = await self._get(key)

        if result:
            record_cache_access(hit=True, cache_type="llm")
            logger.debug("LLM cache hit", query_hash=query_hash)
        else:
            record_cache_access(hit=False, cache_type="llm")

        return result

    async def set_llm_response(
        self,
        query: str,
        model: str,
        response: str,
        temperature: float = 0.0,
    ) -> bool:
        """Cache LLM response.

        Only caches deterministic responses (temperature=0).

        Args:
            query: Original query text.
            model: Model name used.
            response: LLM response to cache.
            temperature: Temperature used (only cache if 0).

        Returns:
            True if cached successfully.
        """
        # Only cache deterministic responses per architecture doc
        if temperature > 0:
            logger.debug("Skipping cache for non-deterministic response")
            return False

        query_hash = _hash_text(query)
        key = self._llm_cache_key(query_hash, model)
        return await self._set(key, response, CacheConfig.LLM_RESPONSE_TTL)

    # Embedding Caching

    def _embedding_cache_key(self, text_hash: str, version: str = "v1") -> str:
        """Generate embedding cache key."""
        return f"{CacheConfig.EMBEDDING_PREFIX}:{text_hash}:{version}"

    async def get_embedding(
        self,
        text: str,
        version: str = "v1",
    ) -> list[float] | None:
        """Get cached embedding.

        Args:
            text: Text that was embedded.
            version: Embedding model version.

        Returns:
            Cached embedding vector or None.
        """
        text_hash = _hash_text(text)
        key = self._embedding_cache_key(text_hash, version)
        result = await self._get(key)

        if result:
            record_cache_access(hit=True, cache_type="embedding")
            logger.debug("Embedding cache hit", text_hash=text_hash)
            return json.loads(result)
        else:
            record_cache_access(hit=False, cache_type="embedding")
            return None

    async def set_embedding(
        self,
        text: str,
        embedding: list[float],
        version: str = "v1",
    ) -> bool:
        """Cache embedding.

        Args:
            text: Original text.
            embedding: Embedding vector.
            version: Embedding model version.

        Returns:
            True if cached successfully.
        """
        text_hash = _hash_text(text)
        key = self._embedding_cache_key(text_hash, version)
        return await self._set(
            key, json.dumps(embedding), CacheConfig.EMBEDDING_TTL
        )

    async def invalidate_embedding(
        self,
        text: str,
        version: str = "v1",
    ) -> bool:
        """Invalidate cached embedding.

        Called when documents are re-ingested.

        Args:
            text: Original text.
            version: Embedding model version.

        Returns:
            True if deleted successfully.
        """
        text_hash = _hash_text(text)
        key = self._embedding_cache_key(text_hash, version)
        return await self._delete(key)

    # Session Caching

    def _session_cache_key(self, user_id: str) -> str:
        """Generate session cache key."""
        return f"{CacheConfig.SESSION_PREFIX}:{user_id}"

    async def get_session(self, user_id: str) -> dict | None:
        """Get cached user session.

        Args:
            user_id: User identifier.

        Returns:
            Session data or None.
        """
        key = self._session_cache_key(user_id)
        result = await self._get(key)

        if result:
            record_cache_access(hit=True, cache_type="session")
            return json.loads(result)
        else:
            record_cache_access(hit=False, cache_type="session")
            return None

    async def set_session(
        self,
        user_id: str,
        session_data: dict,
    ) -> bool:
        """Cache user session.

        Args:
            user_id: User identifier.
            session_data: Session data to cache.

        Returns:
            True if cached successfully.
        """
        key = self._session_cache_key(user_id)
        return await self._set(
            key, json.dumps(session_data), CacheConfig.SESSION_TTL
        )

    async def extend_session(self, user_id: str) -> bool:
        """Extend session TTL on activity.

        Args:
            user_id: User identifier.

        Returns:
            True if extended successfully.
        """
        if not self._connected or not self._client:
            return False

        key = self._session_cache_key(user_id)
        try:
            await self._client.expire(
                key, int(CacheConfig.SESSION_TTL.total_seconds())
            )
            return True
        except Exception as e:
            logger.warning("Session extend failed", user_id=user_id, error=str(e))
            return False

    async def delete_session(self, user_id: str) -> bool:
        """Delete user session (logout).

        Args:
            user_id: User identifier.

        Returns:
            True if deleted successfully.
        """
        key = self._session_cache_key(user_id)
        return await self._delete(key)

    # Rate Limiting (sliding window)

    def _rate_limit_key(self, user_id: str, endpoint: str) -> str:
        """Generate rate limit cache key."""
        return f"{CacheConfig.RATE_LIMIT_PREFIX}:{user_id}:{endpoint}"

    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        max_requests: int,
        window_seconds: int = 60,
    ) -> tuple[bool, int]:
        """Check and update rate limit.

        Uses sliding window algorithm.

        Args:
            user_id: User identifier.
            endpoint: API endpoint.
            max_requests: Maximum requests allowed.
            window_seconds: Time window in seconds.

        Returns:
            Tuple of (is_allowed, remaining_requests).
        """
        if not self._connected or not self._client:
            return True, max_requests  # Allow if cache unavailable

        import time

        key = self._rate_limit_key(user_id, endpoint)
        now = time.time()
        window_start = now - window_seconds

        try:
            # Use sorted set for sliding window
            pipe = self._client.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Add current request
            pipe.zadd(key, {str(now): now})

            # Count requests in window
            pipe.zcount(key, window_start, now)

            # Set TTL
            pipe.expire(key, window_seconds)

            results = await pipe.execute()
            current_count = results[2]

            is_allowed = current_count <= max_requests
            remaining = max(0, max_requests - current_count)

            if not is_allowed:
                logger.warning(
                    "Rate limit exceeded",
                    user_id=user_id,
                    endpoint=endpoint,
                    count=current_count,
                )

            return is_allowed, remaining

        except Exception as e:
            logger.warning("Rate limit check failed", error=str(e))
            return True, max_requests  # Fail open


# Singleton cache instance
_cache_instance: RedisCache | None = None


async def get_cache() -> RedisCache:
    """Get or create the Redis cache instance.

    Returns:
        Configured RedisCache instance.
    """
    global _cache_instance

    if _cache_instance is None:
        settings = get_settings()
        _cache_instance = RedisCache(
            host=settings.redis.host,
            port=settings.redis.port,
            password=settings.redis.password,
            db=settings.redis.db,
            ssl=settings.redis.ssl,
        )
        await _cache_instance.connect()

    return _cache_instance


async def close_cache() -> None:
    """Close the cache connection."""
    global _cache_instance
    if _cache_instance:
        await _cache_instance.disconnect()
        _cache_instance = None
