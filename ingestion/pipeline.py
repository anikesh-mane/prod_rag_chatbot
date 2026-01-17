"""Document ingestion pipeline with versioning support."""

from pathlib import Path

import numpy as np
import structlog

from configs import get_settings
from ingestion.chunkers import RecursiveChunker
from ingestion.embedders import get_embedder
from ingestion.loaders import load_document
from ingestion.versioning import get_version_manager
from monitoring.metrics import record_ingestion
from schemas import Chunk, Document

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting documents into the vector store with versioning."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize ingestion pipeline.

        Args:
            chunk_size: Override default chunk size.
            chunk_overlap: Override default chunk overlap.
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.ingestion.chunk_size
        self.chunk_overlap = chunk_overlap or settings.ingestion.chunk_overlap

        self.chunker = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.embedder = get_embedder()

        # Get embedding version info
        self._version_manager = get_version_manager()
        self._current_version = self._version_manager.get_active_version()

    async def ingest_file(
        self,
        file_path: str | Path,
    ) -> tuple[Document, list[Chunk], list[np.ndarray]]:
        """Ingest a single file.

        Args:
            file_path: Path to file to ingest.

        Returns:
            Tuple of (document, chunks, embeddings).
        """
        logger.info("Starting file ingestion", path=str(file_path))

        # Load document
        document = await load_document(file_path)
        logger.debug("Document loaded", doc_id=str(document.document_id))

        # Chunk document
        chunks = self.chunker.chunk(document)
        logger.debug("Document chunked", chunk_count=len(chunks))

        # Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedder.embed_batch(chunk_texts)
        logger.debug("Embeddings generated", embedding_count=len(embeddings))

        # Link embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding_id = f"{chunk.chunk_id}"

        # Record metrics
        record_ingestion(
            document_count=1,
            chunk_count=len(chunks),
            source_type=document.metadata.get("file_type", "unknown"),
        )

        logger.info(
            "File ingestion complete",
            path=str(file_path),
            doc_id=str(document.document_id),
            chunks=len(chunks),
            embedding_version=self._current_version.version_id if self._current_version else "unknown",
        )

        return document, chunks, embeddings

    async def ingest_text(
        self,
        content: str,
        source: str,
        metadata: dict | None = None,
    ) -> tuple[Document, list[Chunk], list[np.ndarray]]:
        """Ingest raw text content.

        Args:
            content: Text content to ingest.
            source: Source identifier.
            metadata: Optional metadata.

        Returns:
            Tuple of (document, chunks, embeddings).
        """
        logger.info("Starting text ingestion", source=source)

        # Create document
        document = Document(
            source=source,
            content=content,
            metadata=metadata or {},
        )

        # Chunk document
        chunks = self.chunker.chunk(document)

        # Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedder.embed_batch(chunk_texts)

        # Link embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding_id = f"{chunk.chunk_id}"

        # Record metrics
        record_ingestion(
            document_count=1,
            chunk_count=len(chunks),
            source_type="text",
        )

        logger.info(
            "Text ingestion complete",
            source=source,
            doc_id=str(document.document_id),
            chunks=len(chunks),
            embedding_version=self._current_version.version_id if self._current_version else "unknown",
        )

        return document, chunks, embeddings

    async def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[tuple[Document, list[Chunk], list[np.ndarray]]]:
        """Ingest all supported files in a directory.

        Args:
            directory: Directory path.
            recursive: Whether to search subdirectories.

        Returns:
            List of (document, chunks, embeddings) tuples.
        """
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        results: list[tuple[Document, list[Chunk], list[np.ndarray]]] = []

        supported_extensions = {".txt", ".md", ".pdf", ".docx"}

        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    result = await self.ingest_file(file_path)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        "Failed to ingest file",
                        path=str(file_path),
                        error=str(e),
                    )

        logger.info(
            "Directory ingestion complete",
            directory=str(directory),
            files_processed=len(results),
        )

        return results
