"""Base text chunker interface."""

from abc import ABC, abstractmethod

from schemas import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize chunker with size parameters.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of chunks with metadata.
        """
        pass
