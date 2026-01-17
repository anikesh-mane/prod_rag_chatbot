"""Document ingestion module."""

from ingestion.pipeline import IngestionPipeline
from ingestion.versioning import (
    EmbeddingVersion,
    EmbeddingVersionManager,
    EmbeddingVersionStatus,
    get_version_manager,
)

__all__ = [
    "IngestionPipeline",
    "EmbeddingVersion",
    "EmbeddingVersionManager",
    "EmbeddingVersionStatus",
    "get_version_manager",
]
