"""Main retrieval module combining vector search and reranking."""

import time

import structlog

from configs import get_settings
from ingestion.embedders import get_embedder
from monitoring.metrics import metrics, record_retrieval
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import MilvusVectorStore
from schemas import RetrievalResult

logger = structlog.get_logger(__name__)


class Retriever:
    """Retriever combining vector search with optional reranking."""

    def __init__(
        self,
        vector_store: MilvusVectorStore | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        rerank_enabled: bool | None = None,
        rerank_model: str | None = None,
    ):
        """Initialize retriever.

        Args:
            vector_store: Vector store instance (created if not provided).
            top_k: Number of results to retrieve.
            score_threshold: Minimum similarity score.
            rerank_enabled: Whether to use reranking.
            rerank_model: Reranking model name.
        """
        settings = get_settings()

        self.top_k = top_k or settings.retrieval.top_k
        self.score_threshold = score_threshold or settings.retrieval.score_threshold
        self.rerank_enabled = (
            rerank_enabled if rerank_enabled is not None else settings.retrieval.rerank_enabled
        )

        self._vector_store = vector_store or MilvusVectorStore()
        self._embedder = get_embedder()
        self._reranker: CrossEncoderReranker | None = None

        if self.rerank_enabled:
            rerank_model = rerank_model or settings.retrieval.rerank_model
            self._reranker = CrossEncoderReranker(model_name=rerank_model)

    async def initialize(self) -> None:
        """Initialize connections."""
        await self._vector_store.connect()
        logger.info("Retriever initialized")

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_expr: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: User query string.
            top_k: Override default top_k.
            filter_expr: Optional metadata filter.

        Returns:
            List of relevant retrieval results.
        """
        top_k = top_k or self.top_k
        start_time = time.perf_counter()
        logger.info("Starting retrieval", query=query[:100], top_k=top_k)

        # Generate query embedding
        query_embedding = await self._embedder.embed(query)

        # Retrieve from vector store
        # If reranking, fetch more candidates
        fetch_k = top_k * 3 if self.rerank_enabled else top_k

        results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            score_threshold=self.score_threshold,
            filter_expr=filter_expr,
        )

        logger.debug("Initial retrieval complete", count=len(results))

        # Apply reranking if enabled
        if self.rerank_enabled and self._reranker and results:
            with metrics.timer("reranking_duration_seconds"):
                results = await self._reranker.rerank(
                    query=query,
                    results=results,
                    top_k=top_k,
                )
            logger.debug("Reranking complete", count=len(results))

        # Ensure we return exactly top_k
        results = results[:top_k]

        # Record metrics
        duration = time.perf_counter() - start_time
        record_retrieval(
            duration=duration,
            result_count=len(results),
            collection=self._vector_store._collection_name,
        )

        logger.info(
            "Retrieval complete",
            query=query[:50],
            results=len(results),
            top_score=results[0].score if results else 0.0,
            duration_ms=round(duration * 1000, 2),
        )

        return results

    async def retrieve_with_metadata(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        metadata_filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve with metadata filtering.

        Args:
            query: User query string.
            top_k: Number of results.
            document_ids: Filter to specific documents.
            metadata_filters: Additional metadata filters.

        Returns:
            Filtered retrieval results.
        """
        filter_parts: list[str] = []

        if document_ids:
            ids_str = ", ".join(f'"{id}"' for id in document_ids)
            filter_parts.append(f"document_id in [{ids_str}]")

        if metadata_filters:
            for key, value in metadata_filters.items():
                if isinstance(value, str):
                    filter_parts.append(f'metadata["{key}"] == "{value}"')
                else:
                    filter_parts.append(f'metadata["{key}"] == {value}')

        filter_expr = " and ".join(filter_parts) if filter_parts else None

        return await self.retrieve(
            query=query,
            top_k=top_k,
            filter_expr=filter_expr,
        )

    async def close(self) -> None:
        """Close connections."""
        await self._vector_store.disconnect()
        logger.info("Retriever closed")
