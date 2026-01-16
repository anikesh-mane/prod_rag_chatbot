"""Text and markdown file loader."""

import aiofiles
from pathlib import Path

import structlog

from ingestion.loaders.base import BaseLoader
from schemas import Document

logger = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}


class TextLoader(BaseLoader):
    """Loader for plain text and markdown files."""

    def supports(self, source: str | Path) -> bool:
        """Check if source is a supported text file."""
        path = Path(source)
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    async def load(self, source: str | Path) -> Document:
        """Load text content from file.

        Args:
            source: Path to text file.

        Returns:
            Document with text content.
        """
        path = Path(source)
        logger.info("Loading text file", path=str(path))

        async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
            content = await f.read()

        return Document(
            source=str(path),
            content=content,
            metadata={
                "file_type": path.suffix.lstrip("."),
                "file_name": path.name,
                "file_size": path.stat().st_size,
            },
        )
