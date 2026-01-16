"""Embedding model implementations."""

from typing import Literal

from configs import get_settings
from ingestion.embedders.base import BaseEmbedder
from ingestion.embedders.openai_embedder import OpenAIEmbedder
from ingestion.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder


def get_embedder(
    provider: Literal["openai", "sentence-transformers"] | None = None,
) -> BaseEmbedder:
    """Get configured embedder instance.

    Args:
        provider: Embedding provider. Uses config default if not specified.

    Returns:
        Configured embedder instance.
    """
    settings = get_settings()
    provider = provider or settings.embedding.provider

    if provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.embedding.model,
            dimension=settings.embedding.dimension,
            batch_size=settings.embedding.batch_size,
        )
    elif provider == "sentence-transformers":
        return SentenceTransformerEmbedder(
            model=settings.embedding.model,
            batch_size=settings.embedding.batch_size,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


__all__ = [
    "BaseEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
]
