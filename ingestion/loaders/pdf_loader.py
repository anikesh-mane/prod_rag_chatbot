"""PDF document loader."""

from pathlib import Path

import structlog
from pypdf import PdfReader

from ingestion.loaders.base import BaseLoader
from schemas import Document

logger = structlog.get_logger(__name__)


class PDFLoader(BaseLoader):
    """Loader for PDF documents."""

    def supports(self, source: str | Path) -> bool:
        """Check if source is a PDF file."""
        path = Path(source)
        return path.suffix.lower() == ".pdf"

    async def load(self, source: str | Path) -> Document:
        """Load text content from PDF.

        Args:
            source: Path to PDF file.

        Returns:
            Document with extracted text content.
        """
        path = Path(source)
        logger.info("Loading PDF file", path=str(path))

        reader = PdfReader(path)
        pages_text: list[str] = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)
            else:
                logger.warning("Empty page in PDF", path=str(path), page=page_num)

        content = "\n\n".join(pages_text)

        return Document(
            source=str(path),
            content=content,
            metadata={
                "file_type": "pdf",
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "page_count": len(reader.pages),
            },
        )
