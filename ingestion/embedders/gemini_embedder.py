"""Gemini embedding implementation with Redis caching."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.embedders.base import BaseEmbedder

if TYPE_CHECKING:
    from retrieval.cache import RedisCache

logger = structlog.get_logger(__name__)


class GeminiEmbedder(BaseEmbedder):
    """Embedder using Google Gemini embedding API with optional Redis caching.

    Uses the unified Google GenAI SDK (google-genai) with async support.
    Default model is gemini-embedding-001 which outputs 768-dimensional embeddings,
    but supports output dimensionality up to 3072 via Matryoshka Representation Learning.
    """

    DEFAULT_MODEL = "gemini-embedding-001"
    DEFAULT_DIMENSION = 768
    MAX_DIMENSION = 3072

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        batch_size: int = 100,
        cache: RedisCache | None = None,
        cache_version: str = "v1",
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        """Initialize Gemini embedder.

        Args:
            api_key: Google Gemini API key.
            model: Embedding model name (default: gemini-embedding-001).
            dimension: Output embedding dimension (default: 768, max: 3072).
            batch_size: Maximum texts per API call.
            cache: Optional Redis cache for embedding caching.
            cache_version: Version string for cache keys.
            task_type: Task type for embeddings. Options:
                - RETRIEVAL_DOCUMENT: For document indexing
                - RETRIEVAL_QUERY: For query embeddings
                - SEMANTIC_SIMILARITY: For similarity comparison
                - CLASSIFICATION: For text classification
                - CLUSTERING: For text clustering
        """
        if dimension > self.MAX_DIMENSION:
            raise ValueError(
                f"Dimension {dimension} exceeds maximum {self.MAX_DIMENSION} "
                f"for Gemini embeddings"
            )

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._cache = cache
        self._cache_version = cache_version
        self._task_type = task_type

        logger.info(
            "Initialized Gemini embedder",
            model=model,
            dimension=dimension,
            task_type=task_type,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Uses Redis cache if available.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        # Check cache first
        if self._cache and self._cache.is_connected():
            cached = await self._cache.get_embedding(text, self._cache_version)
            if cached is not None:
                logger.debug("Embedding cache hit", text_len=len(text))
                return np.array(cached, dtype=np.float32)

        # Generate embedding using async client
        config = types.EmbedContentConfig(
            task_type=self._task_type,
            output_dimensionality=self._dimension,
        )

        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
            config=config,
        )

        # Extract embedding from response
        embedding = response.embeddings[0].values

        # Cache the result
        if self._cache and self._cache.is_connected():
            await self._cache.set_embedding(text, embedding, self._cache_version)

        return np.array(embedding, dtype=np.float32)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        all_embeddings: list[np.ndarray] = []
        config = types.EmbedContentConfig(
            task_type=self._task_type,
            output_dimensionality=self._dimension,
        )

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug(
                "Embedding batch",
                batch_start=i,
                batch_size=len(batch),
                total=len(texts),
            )

            # Gemini supports batch embedding via contents list
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=batch,
                config=config,
            )

            # Extract embeddings - response.embeddings is a list
            batch_embeddings = [
                np.array(emb.values, dtype=np.float32) for emb in response.embeddings
            ]
            all_embeddings.extend(batch_embeddings)

        logger.info(
            "Batch embedding complete",
            total_texts=len(texts),
            total_embeddings=len(all_embeddings),
        )
        return all_embeddings
