"""Document loaders for various file types."""

from pathlib import Path

from ingestion.loaders.base import BaseLoader
from ingestion.loaders.docx_loader import DocxLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.text_loader import TextLoader
from schemas import Document

# Registry of available loaders
_LOADERS: list[BaseLoader] = [
    TextLoader(),
    PDFLoader(),
    DocxLoader(),
]


def get_loader(source: str | Path) -> BaseLoader:
    """Get appropriate loader for the given source.

    Args:
        source: File path to load.

    Returns:
        Loader instance that supports the source.

    Raises:
        ValueError: If no loader supports the source.
    """
    for loader in _LOADERS:
        if loader.supports(source):
            return loader
    raise ValueError(f"No loader available for: {source}")


async def load_document(source: str | Path) -> Document:
    """Load a document using the appropriate loader.

    Args:
        source: File path to load.

    Returns:
        Loaded document.
    """
    loader = get_loader(source)
    return await loader.load(source)


__all__ = [
    "BaseLoader",
    "TextLoader",
    "PDFLoader",
    "DocxLoader",
    "get_loader",
    "load_document",
]
