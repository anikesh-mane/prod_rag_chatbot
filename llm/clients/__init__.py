"""LLM client implementations."""

from configs import get_settings
from llm.clients.base import BaseLLMClient
from llm.clients.fallback_client import FallbackLLMClient
from llm.clients.gemini_client import GeminiClient
from llm.clients.openai_client import OpenAIClient


def get_llm_client(with_fallback: bool = True) -> BaseLLMClient:
    """Get configured LLM client.

    Args:
        with_fallback: Whether to wrap with fallback client.

    Returns:
        Configured LLM client.
    """
    settings = get_settings()
    provider = settings.llm.provider

    if provider == "gemini":
        primary = GeminiClient(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.llm.model,
            timeout=settings.llm.timeout,
            max_retries=settings.llm.max_retries,
        )

        if not with_fallback:
            return primary

        # Gemini fallback uses a different model in the same family
        fallback = GeminiClient(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.llm.fallback_model,
            timeout=settings.llm.timeout,
            max_retries=settings.llm.max_retries,
        )

        return FallbackLLMClient(primary=primary, fallback=fallback)

    # Default to OpenAI
    primary = OpenAIClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.llm.model,
        timeout=settings.llm.timeout,
        max_retries=settings.llm.max_retries,
    )

    if not with_fallback:
        return primary

    fallback = OpenAIClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.llm.fallback_model,
        timeout=settings.llm.timeout,
        max_retries=settings.llm.max_retries,
    )

    return FallbackLLMClient(primary=primary, fallback=fallback)


__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "GeminiClient",
    "FallbackLLMClient",
    "get_llm_client",
]
