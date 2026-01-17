"""LLM client with fallback support."""

from typing import AsyncGenerator

import structlog

from llm.clients.base import BaseLLMClient
from schemas import LLMRequest, LLMResponse

logger = structlog.get_logger(__name__)


class FallbackLLMClient(BaseLLMClient):
    """LLM client that falls back to secondary models on failure."""

    def __init__(
        self,
        primary: BaseLLMClient,
        fallback: BaseLLMClient,
    ):
        """Initialize fallback client.

        Args:
            primary: Primary LLM client.
            fallback: Fallback LLM client to use on primary failure.
        """
        self._primary = primary
        self._fallback = fallback

    @property
    def model_name(self) -> str:
        return f"{self._primary.model_name} (fallback: {self._fallback.model_name})"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate with fallback on failure.

        Args:
            request: LLM request.

        Returns:
            LLM response from primary or fallback.
        """
        try:
            logger.debug("Attempting primary LLM", model=self._primary.model_name)
            return await self._primary.generate(request)
        except Exception as e:
            logger.warning(
                "Primary LLM failed, using fallback",
                primary_model=self._primary.model_name,
                fallback_model=self._fallback.model_name,
                error=str(e),
            )
            return await self._fallback.generate(request)

    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Generate stream with fallback on failure.

        Args:
            request: LLM request.

        Yields:
            Response content chunks.
        """
        try:
            logger.debug("Attempting primary LLM stream", model=self._primary.model_name)
            async for chunk in self._primary.generate_stream(request):
                yield chunk
        except Exception as e:
            logger.warning(
                "Primary LLM stream failed, using fallback",
                primary_model=self._primary.model_name,
                fallback_model=self._fallback.model_name,
                error=str(e),
            )
            async for chunk in self._fallback.generate_stream(request):
                yield chunk

    async def health_check(self) -> bool:
        """Check if at least one client is healthy.

        Returns:
            True if primary or fallback is healthy.
        """
        primary_healthy = await self._primary.health_check()
        if primary_healthy:
            return True

        logger.warning("Primary LLM unhealthy, checking fallback")
        return await self._fallback.health_check()
