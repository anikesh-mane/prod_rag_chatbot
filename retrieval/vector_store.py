"""Milvus vector store implementation."""

from typing import Any

import numpy as np
import structlog
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from configs import get_settings
from schemas import RetrievalResult

logger = structlog.get_logger(__name__)


class MilvusVectorStore:
    """Vector store using Milvus for similarity search."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        dimension: int | None = None,
    ):
        """Initialize Milvus vector store.

        Args:
            host: Milvus server host.
            port: Milvus server port.
            collection_name: Name of the collection.
            dimension: Embedding dimension.
        """
        settings = get_settings()
        self.host = host or settings.milvus.host
        self.port = port or settings.milvus.port
        self.collection_name = collection_name or settings.milvus.collection_name
        self.dimension = dimension or settings.embedding.dimension
        self.index_type = settings.milvus.index_type
        self.metric_type = settings.milvus.metric_type
        self.nlist = settings.milvus.nlist
        self.nprobe = settings.milvus.nprobe

        self._collection: Collection | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Milvus server and initialize collection."""
        if self._connected:
            return

        logger.info(
            "Connecting to Milvus",
            host=self.host,
            port=self.port,
        )

        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
        )

        self._ensure_collection()
        self._connected = True
        logger.info("Connected to Milvus", collection=self.collection_name)

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            self._collection.load()
            logger.debug("Loaded existing collection", name=self.collection_name)
            return

        # Define schema
        fields = [
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=64,
            ),
            FieldSchema(
                name="document_id",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="RAG document chunks with embeddings",
        )

        # Create collection
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        # Create index on embedding field
        index_params = {
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "params": {"nlist": self.nlist},
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )

        self._collection.load()
        logger.info("Created new collection", name=self.collection_name)

    async def insert(
        self,
        chunk_ids: list[str],
        document_ids: list[str],
        contents: list[str],
        embeddings: list[np.ndarray],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Insert chunks with embeddings into the vector store.

        Args:
            chunk_ids: List of chunk IDs.
            document_ids: List of document IDs.
            contents: List of text contents.
            embeddings: List of embedding vectors.
            metadata: List of metadata dicts.

        Returns:
            Number of inserted entities.
        """
        if not self._collection:
            raise RuntimeError("Not connected to Milvus")

        # Convert embeddings to list format
        embedding_lists = [emb.tolist() for emb in embeddings]

        data = [
            chunk_ids,
            document_ids,
            contents,
            embedding_lists,
            metadata,
        ]

        result = self._collection.insert(data)
        self._collection.flush()

        logger.info(
            "Inserted chunks into Milvus",
            count=result.insert_count,
        )
        return result.insert_count

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_expr: str | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.
            filter_expr: Optional Milvus filter expression.

        Returns:
            List of retrieval results.
        """
        if not self._collection:
            raise RuntimeError("Not connected to Milvus")

        search_params = {
            "metric_type": self.metric_type,
            "params": {"nprobe": self.nprobe},
        }

        results = self._collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["chunk_id", "document_id", "content", "metadata"],
        )

        retrieval_results: list[RetrievalResult] = []

        for hits in results:
            for hit in hits:
                # Convert distance to similarity score
                # For COSINE, distance is already similarity (0-1)
                # For L2, we need to convert
                if self.metric_type == "COSINE":
                    score = hit.distance
                else:
                    # L2 distance: lower is better, convert to similarity
                    score = 1.0 / (1.0 + hit.distance)

                if score < score_threshold:
                    continue

                retrieval_results.append(
                    RetrievalResult(
                        chunk_id=hit.entity.get("chunk_id"),
                        document_id=hit.entity.get("document_id"),
                        content=hit.entity.get("content"),
                        score=score,
                        metadata=hit.entity.get("metadata", {}),
                    )
                )

        logger.debug(
            "Milvus search complete",
            results=len(retrieval_results),
            top_k=top_k,
        )
        return retrieval_results

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Document ID to delete.

        Returns:
            Number of deleted entities.
        """
        if not self._collection:
            raise RuntimeError("Not connected to Milvus")

        expr = f'document_id == "{document_id}"'
        result = self._collection.delete(expr)
        self._collection.flush()

        logger.info("Deleted chunks", document_id=document_id)
        return result.delete_count

    async def get_stats(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with collection stats.
        """
        if not self._collection:
            raise RuntimeError("Not connected to Milvus")

        return {
            "collection_name": self.collection_name,
            "num_entities": self._collection.num_entities,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
        }

    async def disconnect(self) -> None:
        """Disconnect from Milvus."""
        if self._connected:
            connections.disconnect("default")
            self._connected = False
            self._collection = None
            logger.info("Disconnected from Milvus")

    async def health_check(self) -> bool:
        """Check if Milvus connection is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            if not self._connected:
                return False
            # Try to get collection stats
            _ = self._collection.num_entities
            return True
        except Exception as e:
            logger.error("Milvus health check failed", error=str(e))
            return False
