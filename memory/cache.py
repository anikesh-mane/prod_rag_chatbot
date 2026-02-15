"""Redis caching layer for conversation memory."""

from datetime import datetime, timedelta
import json

import structlog

from retrieval.cache import RedisCache

logger = structlog.get_logger(__name__)


class CacheConfig:
    """Cache configuration for conversations."""

    # TTLs
    CONVERSATION_TTL = timedelta(minutes=30)

    # Key prefixes following pattern: {service}:{entity_type}:{identifier}
    CONVERSATION_PREFIX = "rag:conversation"

    # Limits
    DEFAULT_MAX_MESSAGES = 20


class ConversationCache:
    """Redis cache operations for conversation memory.

    Key pattern: rag:conversation:{conversation_id}:messages
    Data structure: Redis List (LPUSH/LRANGE for efficient recent access)

    Strategy:
    - Store recent messages in Redis for fast access
    - TTL extends on activity (30 min default)
    - Returns None on cache miss to trigger PostgreSQL fallback
    """

    def __init__(self, redis_cache: RedisCache, ttl_minutes: int = 30):
        """Initialize with existing Redis cache.

        Args:
            redis_cache: Configured RedisCache instance.
            ttl_minutes: Conversation TTL in minutes.
        """
        self._cache = redis_cache
        self._ttl = timedelta(minutes=ttl_minutes)

    def _messages_key(self, conversation_id: str) -> str:
        """Generate cache key for conversation messages."""
        return f"{CacheConfig.CONVERSATION_PREFIX}:{conversation_id}:messages"

    def _meta_key(self, conversation_id: str) -> str:
        """Generate cache key for conversation metadata."""
        return f"{CacheConfig.CONVERSATION_PREFIX}:{conversation_id}:meta"

    async def init_conversation(self, conversation_id: str) -> bool:
        """Initialize cache entry for new conversation.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if initialized successfully.
        """
        if not self._cache.is_connected():
            return False

        try:
            key = self._meta_key(conversation_id)
            meta = {
                "created_at": datetime.utcnow().isoformat(),
                "message_count": 0,
            }
            await self._cache._client.setex(
                key,
                int(self._ttl.total_seconds()),
                json.dumps(meta),
            )
            return True
        except Exception as e:
            logger.warning("Cache init failed", error=str(e))
            return False

    async def add_message(
        self,
        conversation_id: str,
        message: dict,
        max_messages: int = CacheConfig.DEFAULT_MAX_MESSAGES,
    ) -> bool:
        """Add message to conversation cache.

        Uses LPUSH + LTRIM for bounded list.

        Args:
            conversation_id: Conversation identifier.
            message: Message dict {role, content, id, created_at}.
            max_messages: Maximum messages to retain.

        Returns:
            True if added successfully.
        """
        if not self._cache.is_connected():
            return False

        try:
            key = self._messages_key(conversation_id)

            pipe = self._cache._client.pipeline()

            # Add to front of list (most recent first)
            pipe.lpush(key, json.dumps(message))

            # Trim to max size
            pipe.ltrim(key, 0, max_messages - 1)

            # Extend TTL
            pipe.expire(key, int(self._ttl.total_seconds()))

            await pipe.execute()

            logger.debug(
                "Message cached",
                conversation_id=conversation_id,
                role=message.get("role"),
            )
            return True

        except Exception as e:
            logger.warning("Cache add_message failed", error=str(e))
            return False

    async def get_messages(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[dict] | None:
        """Get recent messages from cache.

        Args:
            conversation_id: Conversation identifier.
            limit: Maximum messages to return.

        Returns:
            List of messages (oldest to newest), or None on cache miss.
        """
        if not self._cache.is_connected():
            return None

        try:
            key = self._messages_key(conversation_id)
            limit = limit or CacheConfig.DEFAULT_MAX_MESSAGES

            # Check if key exists first
            exists = await self._cache._client.exists(key)
            if not exists:
                return None  # Cache miss - trigger PostgreSQL fallback

            # Get messages (stored newest-first, need to reverse)
            raw_messages = await self._cache._client.lrange(key, 0, limit - 1)

            if not raw_messages:
                return None

            # Parse and reverse to chronological order
            messages = [json.loads(msg) for msg in raw_messages]
            messages.reverse()

            # Extend TTL on access
            await self._cache._client.expire(key, int(self._ttl.total_seconds()))

            return messages

        except Exception as e:
            logger.warning("Cache get_messages failed", error=str(e))
            return None

    async def populate_from_db(
        self,
        conversation_id: str,
        messages: list[dict],
    ) -> bool:
        """Populate cache from database (cache miss scenario).

        Args:
            conversation_id: Conversation identifier.
            messages: Messages to cache (chronological order).

        Returns:
            True if populated successfully.
        """
        if not self._cache.is_connected() or not messages:
            return False

        try:
            key = self._messages_key(conversation_id)

            # Delete existing (if any)
            await self._cache._client.delete(key)

            # Add messages in reverse order (newest first for LPUSH)
            pipe = self._cache._client.pipeline()
            for msg in reversed(messages):
                pipe.lpush(key, json.dumps(msg))

            pipe.expire(key, int(self._ttl.total_seconds()))
            await pipe.execute()

            logger.debug(
                "Cache populated from DB",
                conversation_id=conversation_id,
                message_count=len(messages),
            )
            return True

        except Exception as e:
            logger.warning("Cache populate failed", error=str(e))
            return False

    async def extend_ttl(self, conversation_id: str) -> bool:
        """Extend conversation TTL on activity.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if extended successfully.
        """
        if not self._cache.is_connected():
            return False

        try:
            messages_key = self._messages_key(conversation_id)
            meta_key = self._meta_key(conversation_id)
            ttl_seconds = int(self._ttl.total_seconds())

            pipe = self._cache._client.pipeline()
            pipe.expire(messages_key, ttl_seconds)
            pipe.expire(meta_key, ttl_seconds)
            await pipe.execute()

            return True
        except Exception as e:
            logger.warning("TTL extend failed", error=str(e))
            return False

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Remove conversation from cache.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if deleted successfully.
        """
        if not self._cache.is_connected():
            return False

        try:
            messages_key = self._messages_key(conversation_id)
            meta_key = self._meta_key(conversation_id)

            await self._cache._client.delete(messages_key, meta_key)

            logger.debug(
                "Conversation deleted from cache",
                conversation_id=conversation_id,
            )
            return True
        except Exception as e:
            logger.warning("Cache delete failed", error=str(e))
            return False
