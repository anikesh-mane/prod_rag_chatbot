"""Recursive text chunker that respects document structure."""

import re

import structlog

from ingestion.chunkers.base import BaseChunker
from schemas import Chunk, Document

logger = structlog.get_logger(__name__)

# Separators ordered by priority (paragraph > sentence > word)
DEFAULT_SEPARATORS = [
    "\n\n",  # Paragraph break
    "\n",  # Line break
    ". ",  # Sentence end
    "? ",  # Question end
    "! ",  # Exclamation end
    "; ",  # Semicolon
    ", ",  # Comma
    " ",  # Word boundary
    "",  # Character level (fallback)
]


class RecursiveChunker(BaseChunker):
    """Chunker that recursively splits text using multiple separators.

    Tries to keep semantically related content together by first splitting
    on larger boundaries (paragraphs), then progressively smaller ones.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        """Initialize recursive chunker.

        Args:
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.
            separators: Custom separator list (default: paragraph > sentence > word).
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks recursively.

        Args:
            document: Document to chunk.

        Returns:
            List of chunks preserving document metadata.
        """
        text = document.content.strip()
        if not text:
            logger.warning("Empty document content", source=document.source)
            return []

        chunks_text = self._split_text(text, self.separators)
        chunks: list[Chunk] = []

        for idx, chunk_text in enumerate(chunks_text):
            chunk = Chunk(
                document_id=document.document_id,
                content=chunk_text,
                index=idx,
                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "chunk_size": len(chunk_text),
                },
            )
            chunks.append(chunk)

        logger.info(
            "Document chunked",
            source=document.source,
            total_chunks=len(chunks),
            avg_chunk_size=sum(len(c.content) for c in chunks) // max(len(chunks), 1),
        )
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators.

        Args:
            text: Text to split.
            separators: Remaining separators to try.

        Returns:
            List of text chunks.
        """
        if not separators:
            # No separators left, split by chunk_size
            return self._split_by_size(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split
            return self._split_by_size(text)

        # Split on current separator
        if separator in text:
            splits = text.split(separator)
        else:
            # Separator not found, try next one
            return self._split_text(text, remaining_separators)

        # Merge small splits and recursively process large ones
        chunks: list[str] = []
        current_chunk = ""

        for split in splits:
            split = split.strip()
            if not split:
                continue

            # Check if adding this split exceeds chunk size
            potential_chunk = (
                f"{current_chunk}{separator}{split}" if current_chunk else split
            )

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)

                # Check if split itself is too large
                if len(split) > self.chunk_size:
                    # Recursively split with remaining separators
                    sub_chunks = self._split_text(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _split_by_size(self, text: str) -> list[str]:
        """Split text into fixed-size chunks.

        Args:
            text: Text to split.

        Returns:
            List of fixed-size chunks.
        """
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap between consecutive chunks.

        Args:
            chunks: List of chunks.

        Returns:
            Chunks with overlap applied.
        """
        if len(chunks) <= 1:
            return chunks

        overlapped: list[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap :] if len(prev_chunk) > self.chunk_overlap else prev_chunk

            # Prepend overlap to current chunk
            overlapped_chunk = f"{overlap_text} {current_chunk}".strip()
            overlapped.append(overlapped_chunk)

        return overlapped
