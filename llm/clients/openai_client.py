"""OpenAI LLM client implementation with Redis caching."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, AsyncGenerator

import structlog
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from llm.clients.base import BaseLLMClient
from monitoring.cost_tracker import cost_tracker
from monitoring.metrics import record_llm_call, record_llm_error, record_cache_access
from schemas import LLMRequest, LLMResponse

if TYPE_CHECKING:
    from retrieval.cache import RedisCache

logger = structlog.get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """LLM client using OpenAI API with optional Redis caching."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        timeout: int = 30,
        max_retries: int = 3,
        cache: RedisCache | None = None,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key.
            model: Model name to use.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            cache: Optional Redis cache for response caching.
        """
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_retries = max_retries
        self._cache = cache

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    )
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from OpenAI.

        Uses Redis cache for deterministic (temperature=0) requests.

        Args:
            request: LLM request.

        Returns:
            LLM response with content and metadata.
        """
        start_time = time.perf_counter()
        model = request.model or self._model

        # Check cache for deterministic requests
        if (
            self._cache
            and self._cache.is_connected()
            and request.temperature == 0.0
        ):
            cached_response = await self._cache.get_llm_response(
                query=request.prompt,
                model=model,
            )
            if cached_response:
                record_cache_access(hit=True, cache_type="llm")
                logger.debug("LLM cache hit", model=model)
                latency_ms = (time.perf_counter() - start_time) * 1000
                return LLMResponse(
                    content=cached_response,
                    model=model,
                    prompt_tokens=0,  # Unknown for cached response
                    completion_tokens=0,
                    total_tokens=0,
                    finish_reason="cached",
                    latency_ms=latency_ms,
                )
            record_cache_access(hit=False, cache_type="llm")

        logger.debug(
            "Calling OpenAI",
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences or None,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            choice = response.choices[0]

            result = LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                finish_reason=choice.finish_reason or "unknown",
                latency_ms=latency_ms,
            )

            # Record metrics and cost
            usage_record = cost_tracker.record_usage(
                model=response.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
            record_llm_call(
                duration=latency_ms / 1000,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                model=response.model,
                cost=usage_record.cost_usd,
            )

            logger.info(
                "OpenAI response received",
                model=model,
                tokens=result.total_tokens,
                latency_ms=round(latency_ms, 2),
                cost_usd=round(usage_record.cost_usd, 6),
            )

            # Cache deterministic responses
            if (
                self._cache
                and self._cache.is_connected()
                and request.temperature == 0.0
            ):
                await self._cache.set_llm_response(
                    query=request.prompt,
                    model=model,
                    response=result.content,
                    temperature=request.temperature,
                )

            return result

        except RateLimitError as e:
            logger.warning("OpenAI rate limit hit", error=str(e))
            record_llm_error(model, "rate_limit")
            raise
        except APITimeoutError as e:
            logger.warning("OpenAI timeout", error=str(e))
            record_llm_error(model, "timeout")
            raise
        except APIError as e:
            logger.error("OpenAI API error", error=str(e))
            record_llm_error(model, "api_error")
            raise

    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from OpenAI.

        Args:
            request: LLM request.

        Yields:
            Response content chunks.
        """
        model = request.model or self._model

        logger.debug("Starting OpenAI stream", model=model)

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences or None,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("OpenAI stream error", error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check OpenAI API health.

        Returns:
            True if API is accessible.
        """
        try:
            # Make a minimal API call
            await self._client.models.retrieve(self._model)
            return True
        except Exception as e:
            logger.error("OpenAI health check failed", error=str(e))
            return False
