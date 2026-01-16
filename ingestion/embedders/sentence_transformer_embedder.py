"""Sentence Transformers embedding implementation."""

import asyncio
from functools import partial

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from ingestion.embedders.base import BaseEmbedder

logger = structlog.get_logger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using Sentence Transformers (local models)."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """Initialize Sentence Transformer embedder.

        Args:
            model: Model name from HuggingFace hub.
            device: Device to run model on ('cpu', 'cuda', 'mps').
            batch_size: Batch size for encoding.
        """
        logger.info("Loading Sentence Transformer model", model=model, device=device)
        self._model = SentenceTransformer(model, device=device)
        self._model_name = model
        self._batch_size = batch_size
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            partial(self._model.encode, text, convert_to_numpy=True),
        )
        return embedding.astype(np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        logger.debug("Embedding batch with Sentence Transformer", count=len(texts))

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            partial(
                self._model.encode,
                texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )

        result = [emb.astype(np.float32) for emb in embeddings]
        logger.info("Batch embedding complete", total=len(result))
        return result
