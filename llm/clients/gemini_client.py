"""Google Gemini LLM client implementation with Redis caching."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, AsyncGenerator

import structlog
from google import genai
from google.genai import types
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


class GeminiClient(BaseLLMClient):
    """LLM client using Google Gemini API with optional Redis caching."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        timeout: int = 30,
        max_retries: int = 3,
        cache: RedisCache | None = None,
    ):
        """Initialize Gemini client.

        Args:
            api_key: Google API key.
            model: Model name to use.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            cache: Optional Redis cache for response caching.
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._cache = cache

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from Gemini.

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
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    finish_reason="cached",
                    latency_ms=latency_ms,
                )
            record_cache_access(hit=False, cache_type="llm")

        logger.debug(
            "Calling Gemini",
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        try:
            # Build generation config
            config = types.GenerateContentConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                stop_sequences=request.stop_sequences or None,
            )

            # Generate response
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=request.prompt,
                config=config,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract content
            content = ""
            if response.text:
                content = response.text

            # Extract token counts from usage metadata
            prompt_tokens = 0
            completion_tokens = 0
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count or 0
                completion_tokens = response.usage_metadata.candidates_token_count or 0

            total_tokens = prompt_tokens + completion_tokens

            # Determine finish reason
            finish_reason = "stop"
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason = response.candidates[0].finish_reason.name.lower()

            result = LLMResponse(
                content=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
            )

            # Record metrics and cost
            usage_record = cost_tracker.record_usage(
                model=model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
            record_llm_call(
                duration=latency_ms / 1000,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                model=model,
                cost=usage_record.cost_usd,
            )

            logger.info(
                "Gemini response received",
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

        except TimeoutError as e:
            logger.warning("Gemini timeout", error=str(e))
            record_llm_error(model, "timeout")
            raise
        except Exception as e:
            logger.error("Gemini API error", error=str(e))
            record_llm_error(model, "api_error")
            raise

    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Gemini.

        Args:
            request: LLM request.

        Yields:
            Response content chunks.
        """
        model = request.model or self._model

        logger.debug("Starting Gemini stream", model=model)

        try:
            # Build generation config
            config = types.GenerateContentConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                stop_sequences=request.stop_sequences or None,
            )

            # Generate streaming response
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=model,
                contents=request.prompt,
                config=config,
            ):
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error("Gemini stream error", error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check Gemini API health.

        Returns:
            True if API is accessible.
        """
        try:
            # List models to verify API access
            models = await self._client.aio.models.list()
            # Check if our model is available
            model_names = [m.name for m in models]
            return any(self._model in name for name in model_names)
        except Exception as e:
            logger.error("Gemini health check failed", error=str(e))
            return False
