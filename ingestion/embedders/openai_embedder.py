"""OpenAI embedding implementation."""

import numpy as np
import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.embedders.base import BaseEmbedder

logger = structlog.get_logger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        batch_size: int = 100,
    ):
        """Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key.
            model: Embedding model name.
            dimension: Expected embedding dimension.
            batch_size: Maximum texts per API call.
        """
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        embedding = response.data[0].embedding
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

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug(
                "Embedding batch",
                batch_start=i,
                batch_size=len(batch),
                total=len(texts),
            )

            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [
                np.array(item.embedding, dtype=np.float32) for item in sorted_data
            ]
            all_embeddings.extend(batch_embeddings)

        logger.info(
            "Batch embedding complete",
            total_texts=len(texts),
            total_embeddings=len(all_embeddings),
        )
        return all_embeddings
