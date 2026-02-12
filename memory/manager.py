"""Conversation memory manager - orchestrates Redis and PostgreSQL."""

from datetime import datetime
from uuid import UUID, uuid4

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db_session
from memory.cache import ConversationCache
from memory.models import Conversation, Message, MessageRole
from memory.schemas import ConversationContext, ConversationResponse

logger = structlog.get_logger(__name__)


class ConversationManager:
    """Manages conversational memory across Redis (hot) and PostgreSQL (cold).

    Strategy:
    - Redis: Stores recent messages for active conversations (fast access)
    - PostgreSQL: Stores full conversation history (durable, queryable)

    On read:
    - Try Redis first (active sessions)
    - On cache miss (old conversation restart): fetch from PostgreSQL with window_size limit

    On write: Write to both (Redis for fast access, PostgreSQL for durability)
    """

    def __init__(
        self,
        cache: ConversationCache | None = None,
        window_size: int = 10,
        max_cache_messages: int = 20,
    ):
        """Initialize conversation manager.

        Args:
            cache: Redis cache instance (optional, for graceful degradation).
            window_size: Max messages to load from PostgreSQL on cache miss.
            max_cache_messages: Max messages to keep in Redis cache.
        """
        self._cache = cache
        self.window_size = window_size
        self.max_cache_messages = max_cache_messages

    # =========================================================================
    # Conversation Lifecycle
    # =========================================================================

    async def create_conversation(
        self,
        user_id: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> UUID:
        """Create a new conversation.

        Args:
            user_id: Optional user identifier.
            title: Optional conversation title.
            metadata: Optional metadata.

        Returns:
            New conversation UUID.
        """
        conversation_id = uuid4()

        async with get_db_session() as session:
            conversation = Conversation(
                id=conversation_id,
                user_id=user_id,
                title=title,
                metadata_=metadata or {},
            )
            session.add(conversation)

        # Initialize empty cache entry
        if self._cache:
            await self._cache.init_conversation(str(conversation_id))

        logger.info(
            "Conversation created",
            conversation_id=str(conversation_id),
            user_id=user_id,
        )

        return conversation_id

    async def get_or_create_conversation(
        self,
        conversation_id: str | None,
        user_id: str | None = None,
    ) -> UUID:
        """Get existing conversation or create new one.

        Args:
            conversation_id: Existing conversation ID (or None to create).
            user_id: User ID for new conversation.

        Returns:
            Conversation UUID (existing or new).
        """
        if conversation_id:
            try:
                conv_uuid = UUID(conversation_id)
                # Verify it exists
                if await self._conversation_exists(conv_uuid):
                    # Extend Redis TTL on access
                    if self._cache:
                        await self._cache.extend_ttl(conversation_id)
                    return conv_uuid
            except ValueError:
                logger.warning(
                    "Invalid conversation_id format",
                    conversation_id=conversation_id,
                )

        # Create new conversation
        return await self.create_conversation(user_id=user_id)

    async def _conversation_exists(self, conversation_id: UUID) -> bool:
        """Check if conversation exists in database."""
        async with get_db_session() as session:
            result = await session.execute(
                select(Conversation.id).where(
                    Conversation.id == conversation_id,
                    Conversation.is_active == True,
                )
            )
            return result.scalar_one_or_none() is not None

    # =========================================================================
    # Message Operations
    # =========================================================================

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        query_id: UUID | None = None,
        sources: list | None = None,
        tokens_used: int | None = None,
        model: str | None = None,
    ) -> UUID:
        """Add a message to the conversation.

        Writes to both PostgreSQL (durable) and Redis (fast access).

        Args:
            conversation_id: Conversation to add message to.
            role: Message role (user/assistant/system).
            content: Message content.
            query_id: Optional query ID (for assistant messages).
            sources: Optional source documents (for assistant messages).
            tokens_used: Optional token count.
            model: Optional model name.

        Returns:
            New message UUID.
        """
        message_id = uuid4()

        async with get_db_session() as session:
            # Get next sequence number
            result = await session.execute(
                select(func.coalesce(func.max(Message.sequence_number), 0)).where(
                    Message.conversation_id == conversation_id
                )
            )
            next_seq = result.scalar_one() + 1

            # Create message
            message = Message(
                id=message_id,
                conversation_id=conversation_id,
                role=MessageRole(role),
                content=content,
                sequence_number=next_seq,
                query_id=query_id,
                sources=sources,
                tokens_used=tokens_used,
                model=model,
            )
            session.add(message)

            # Update conversation timestamp and count
            await session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(
                    updated_at=datetime.utcnow(),
                    message_count=Conversation.message_count + 1,
                )
            )

        # Update Redis cache
        if self._cache:
            await self._cache.add_message(
                conversation_id=str(conversation_id),
                message={
                    "id": str(message_id),
                    "role": role,
                    "content": content,
                    "created_at": datetime.utcnow().isoformat(),
                },
                max_messages=self.max_cache_messages,
            )

        logger.debug(
            "Message added",
            conversation_id=str(conversation_id),
            message_id=str(message_id),
            role=role,
        )

        return message_id

    async def add_turn(
        self,
        conversation_id: UUID,
        user_message: str,
        assistant_message: str,
        query_id: UUID | None = None,
        sources: list | None = None,
        tokens_used: int | None = None,
        model: str | None = None,
    ) -> tuple[UUID, UUID]:
        """Add a complete turn (user question + assistant response).

        Args:
            conversation_id: Conversation ID.
            user_message: User's question.
            assistant_message: Assistant's response.
            query_id: Query ID for the response.
            sources: Retrieved sources.
            tokens_used: Tokens used for generation.
            model: Model used.

        Returns:
            Tuple of (user_message_id, assistant_message_id).
        """
        user_msg_id = await self.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
        )

        assistant_msg_id = await self.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_message,
            query_id=query_id,
            sources=sources,
            tokens_used=tokens_used,
            model=model,
        )

        return user_msg_id, assistant_msg_id

    # =========================================================================
    # History Retrieval
    # =========================================================================

    async def get_context(
        self,
        conversation_id: UUID,
    ) -> ConversationContext:
        """Get conversation context for LLM generation.

        Strategy:
        1. Try Redis cache first (active sessions)
        2. On cache miss (restarted conversation):
           - Fetch last window_size messages from PostgreSQL
           - Populate Redis cache for subsequent requests

        Args:
            conversation_id: Conversation ID.

        Returns:
            ConversationContext with formatted history.
        """
        conv_id_str = str(conversation_id)

        # Try Redis first
        if self._cache:
            cached_messages = await self._cache.get_messages(
                conversation_id=conv_id_str,
                limit=self.max_cache_messages,
            )

            if cached_messages is not None:
                logger.debug(
                    "Context loaded from cache",
                    conversation_id=conv_id_str,
                    message_count=len(cached_messages),
                )
                return ConversationContext(
                    conversation_id=conversation_id,
                    messages=[
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in cached_messages
                    ],
                    message_count=len(cached_messages),
                )

        # Cache miss - fetch from PostgreSQL with window_size limit
        async with get_db_session() as session:
            result = await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence_number.desc())
                .limit(self.window_size)
            )
            messages = list(reversed(result.scalars().all()))

            # Populate Redis cache for subsequent requests
            if self._cache and messages:
                await self._cache.populate_from_db(
                    conversation_id=conv_id_str,
                    messages=[
                        {
                            "id": str(msg.id),
                            "role": msg.role.value,
                            "content": msg.content,
                            "created_at": msg.created_at.isoformat(),
                        }
                        for msg in messages
                    ],
                )

            logger.debug(
                "Context loaded from database",
                conversation_id=conv_id_str,
                message_count=len(messages),
            )

            return ConversationContext(
                conversation_id=conversation_id,
                messages=[
                    {"role": msg.role.value, "content": msg.content} for msg in messages
                ],
                message_count=len(messages),
            )

    async def get_full_history(
        self,
        conversation_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get full conversation history from PostgreSQL.

        Args:
            conversation_id: Conversation ID.
            limit: Maximum messages to return.
            offset: Number of messages to skip.

        Returns:
            List of message dictionaries.
        """
        async with get_db_session() as session:
            result = await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence_number)
                .offset(offset)
                .limit(limit)
            )
            messages = result.scalars().all()

            return [
                {
                    "id": str(msg.id),
                    "role": msg.role.value,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                    "query_id": str(msg.query_id) if msg.query_id else None,
                    "tokens_used": msg.tokens_used,
                    "model": msg.model,
                }
                for msg in messages
            ]

    # =========================================================================
    # Cleanup Operations
    # =========================================================================

    async def end_conversation(self, conversation_id: UUID) -> None:
        """Mark conversation as inactive and flush cache.

        Args:
            conversation_id: Conversation to end.
        """
        async with get_db_session() as session:
            await session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(is_active=False, updated_at=datetime.utcnow())
            )

        # Remove from Redis
        if self._cache:
            await self._cache.delete_conversation(str(conversation_id))

        logger.info("Conversation ended", conversation_id=str(conversation_id))

    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 20,
        include_inactive: bool = False,
    ) -> list[ConversationResponse]:
        """Get user's conversations.

        Args:
            user_id: User identifier.
            limit: Maximum conversations to return.
            include_inactive: Include ended conversations.

        Returns:
            List of conversation summaries.
        """
        async with get_db_session() as session:
            query = (
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
            )

            if not include_inactive:
                query = query.where(Conversation.is_active == True)

            result = await session.execute(query)
            conversations = result.scalars().all()

            return [
                ConversationResponse(
                    id=conv.id,
                    user_id=conv.user_id,
                    title=conv.title,
                    message_count=conv.message_count,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    is_active=conv.is_active,
                )
                for conv in conversations
            ]
