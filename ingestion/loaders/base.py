"""Base document loader interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from schemas import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    async def load(self, source: str | Path) -> Document:
        """Load a document from a source.

        Args:
            source: File path or URL to load from.

        Returns:
            Loaded document with content and metadata.
        """
        pass

    @abstractmethod
    def supports(self, source: str | Path) -> bool:
        """Check if this loader supports the given source.

        Args:
            source: File path or URL to check.

        Returns:
            True if this loader can handle the source.
        """
        pass
