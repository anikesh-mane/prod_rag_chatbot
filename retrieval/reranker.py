"""Cross-encoder reranking for retrieval results."""

import asyncio
from functools import partial

import structlog
from sentence_transformers import CrossEncoder

from schemas import RetrievalResult

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """Reranker using cross-encoder models for more accurate relevance scoring."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name.
            device: Device to run model on.
        """
        logger.info("Loading cross-encoder model", model=model_name)
        self._model = CrossEncoder(model_name, device=device)
        self._model_name = model_name

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results using cross-encoder.

        Args:
            query: Original query string.
            results: Initial retrieval results.
            top_k: Number of results to return (default: all).

        Returns:
            Reranked results sorted by relevance.
        """
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]

        # Score pairs using cross-encoder (run in executor)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            partial(self._model.predict, pairs),
        )

        # Update scores and sort
        reranked: list[tuple[float, RetrievalResult]] = []
        for result, score in zip(results, scores):
            updated_result = RetrievalResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                score=float(score),
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "reranked": True,
                },
            )
            reranked.append((float(score), updated_result))

        # Sort by score descending
        reranked.sort(key=lambda x: x[0], reverse=True)

        # Return top_k results
        final_results = [r for _, r in reranked]
        if top_k:
            final_results = final_results[:top_k]

        logger.debug(
            "Reranking complete",
            input_count=len(results),
            output_count=len(final_results),
        )
        return final_results
