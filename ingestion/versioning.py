"""Embedding versioning for zero-downtime migrations.

Supports running multiple embedding versions simultaneously
and migrating documents between versions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class EmbeddingVersionStatus(str, Enum):
    """Status of an embedding version."""

    ACTIVE = "active"  # Currently in use
    MIGRATING = "migrating"  # Being migrated to
    DEPRECATED = "deprecated"  # Being migrated from
    ARCHIVED = "archived"  # No longer in use


@dataclass
class EmbeddingVersion:
    """Represents an embedding model version."""

    version_id: str  # e.g., "v1", "v2"
    model_name: str  # e.g., "text-embedding-3-small"
    dimension: int
    status: EmbeddingVersionStatus
    collection_name: str  # Milvus collection for this version
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "status": self.status.value,
            "collection_name": self.collection_name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EmbeddingVersion:
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            model_name=data["model_name"],
            dimension=data["dimension"],
            status=EmbeddingVersionStatus(data["status"]),
            collection_name=data["collection_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MigrationProgress:
    """Tracks progress of an embedding migration."""

    source_version: str
    target_version: str
    total_documents: int
    migrated_documents: int
    failed_documents: int
    started_at: datetime
    updated_at: datetime
    status: str  # "in_progress", "completed", "failed"
    error_message: str | None = None

    @property
    def progress_percent(self) -> float:
        """Calculate migration progress percentage."""
        if self.total_documents == 0:
            return 100.0
        return (self.migrated_documents / self.total_documents) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source_version": self.source_version,
            "target_version": self.target_version,
            "total_documents": self.total_documents,
            "migrated_documents": self.migrated_documents,
            "failed_documents": self.failed_documents,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
            "progress_percent": self.progress_percent,
            "error_message": self.error_message,
        }


class EmbeddingVersionManager:
    """Manages embedding versions and migrations.

    Supports:
    - Multiple active versions for A/B testing
    - Zero-downtime migrations
    - Version metadata tracking
    """

    def __init__(
        self,
        config_path: str | Path = "configs/embedding_versions.json",
    ):
        """Initialize version manager.

        Args:
            config_path: Path to version configuration file.
        """
        self._config_path = Path(config_path)
        self._versions: dict[str, EmbeddingVersion] = {}
        self._active_version: str | None = None
        self._migrations: dict[str, MigrationProgress] = {}

        self._load_config()

    def _load_config(self) -> None:
        """Load version configuration from file."""
        if not self._config_path.exists():
            logger.info("No embedding version config found, using defaults")
            self._create_default_version()
            return

        try:
            with open(self._config_path) as f:
                data = json.load(f)

            for version_data in data.get("versions", []):
                version = EmbeddingVersion.from_dict(version_data)
                self._versions[version.version_id] = version
                if version.status == EmbeddingVersionStatus.ACTIVE:
                    self._active_version = version.version_id

            logger.info(
                "Loaded embedding versions",
                count=len(self._versions),
                active=self._active_version,
            )

        except Exception as e:
            logger.error("Failed to load version config", error=str(e))
            self._create_default_version()

    def _create_default_version(self) -> None:
        """Create default version configuration."""
        default = EmbeddingVersion(
            version_id="v1",
            model_name="text-embedding-3-small",
            dimension=1536,
            status=EmbeddingVersionStatus.ACTIVE,
            collection_name="rag_chunks_v1",
        )
        self._versions["v1"] = default
        self._active_version = "v1"
        self._save_config()

    def _save_config(self) -> None:
        """Save version configuration to file."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "versions": [v.to_dict() for v in self._versions.values()],
                "active_version": self._active_version,
            }
            with open(self._config_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved version config")
        except Exception as e:
            logger.error("Failed to save version config", error=str(e))

    def get_active_version(self) -> EmbeddingVersion | None:
        """Get the currently active embedding version.

        Returns:
            Active version or None.
        """
        if self._active_version:
            return self._versions.get(self._active_version)
        return None

    def get_version(self, version_id: str) -> EmbeddingVersion | None:
        """Get a specific version.

        Args:
            version_id: Version identifier.

        Returns:
            Version or None if not found.
        """
        return self._versions.get(version_id)

    def list_versions(self) -> list[EmbeddingVersion]:
        """List all versions.

        Returns:
            List of all versions.
        """
        return list(self._versions.values())

    def register_version(
        self,
        version_id: str,
        model_name: str,
        dimension: int,
        collection_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingVersion:
        """Register a new embedding version.

        Args:
            version_id: Unique version identifier.
            model_name: Embedding model name.
            dimension: Embedding dimension.
            collection_name: Milvus collection name.
            metadata: Additional metadata.

        Returns:
            Created version.
        """
        if version_id in self._versions:
            raise ValueError(f"Version {version_id} already exists")

        version = EmbeddingVersion(
            version_id=version_id,
            model_name=model_name,
            dimension=dimension,
            status=EmbeddingVersionStatus.MIGRATING,
            collection_name=collection_name or f"rag_chunks_{version_id}",
            metadata=metadata or {},
        )

        self._versions[version_id] = version
        self._save_config()

        logger.info(
            "Registered new embedding version",
            version_id=version_id,
            model_name=model_name,
        )

        return version

    def activate_version(self, version_id: str) -> None:
        """Activate a version (make it the primary).

        Marks the previous active version as deprecated.

        Args:
            version_id: Version to activate.
        """
        if version_id not in self._versions:
            raise ValueError(f"Version {version_id} not found")

        # Deprecate current active version
        if self._active_version and self._active_version != version_id:
            old_version = self._versions[self._active_version]
            old_version.status = EmbeddingVersionStatus.DEPRECATED

        # Activate new version
        new_version = self._versions[version_id]
        new_version.status = EmbeddingVersionStatus.ACTIVE
        self._active_version = version_id

        self._save_config()

        logger.info(
            "Activated embedding version",
            version_id=version_id,
            previous=self._active_version,
        )

    def archive_version(self, version_id: str) -> None:
        """Archive a deprecated version.

        Args:
            version_id: Version to archive.
        """
        if version_id not in self._versions:
            raise ValueError(f"Version {version_id} not found")

        if version_id == self._active_version:
            raise ValueError("Cannot archive active version")

        version = self._versions[version_id]
        version.status = EmbeddingVersionStatus.ARCHIVED

        self._save_config()

        logger.info("Archived embedding version", version_id=version_id)

    def start_migration(
        self,
        source_version: str,
        target_version: str,
        total_documents: int,
    ) -> MigrationProgress:
        """Start tracking a migration.

        Args:
            source_version: Source version ID.
            target_version: Target version ID.
            total_documents: Total documents to migrate.

        Returns:
            Migration progress tracker.
        """
        migration_id = f"{source_version}_to_{target_version}"
        now = datetime.now(timezone.utc)

        progress = MigrationProgress(
            source_version=source_version,
            target_version=target_version,
            total_documents=total_documents,
            migrated_documents=0,
            failed_documents=0,
            started_at=now,
            updated_at=now,
            status="in_progress",
        )

        self._migrations[migration_id] = progress

        logger.info(
            "Started embedding migration",
            source=source_version,
            target=target_version,
            total=total_documents,
        )

        return progress

    def update_migration_progress(
        self,
        source_version: str,
        target_version: str,
        migrated: int,
        failed: int = 0,
    ) -> MigrationProgress | None:
        """Update migration progress.

        Args:
            source_version: Source version ID.
            target_version: Target version ID.
            migrated: Documents migrated since last update.
            failed: Documents failed since last update.

        Returns:
            Updated progress or None if not found.
        """
        migration_id = f"{source_version}_to_{target_version}"
        progress = self._migrations.get(migration_id)

        if not progress:
            return None

        progress.migrated_documents += migrated
        progress.failed_documents += failed
        progress.updated_at = datetime.now(timezone.utc)

        # Check if complete
        if (
            progress.migrated_documents + progress.failed_documents
            >= progress.total_documents
        ):
            progress.status = "completed" if progress.failed_documents == 0 else "failed"
            logger.info(
                "Migration completed",
                source=source_version,
                target=target_version,
                migrated=progress.migrated_documents,
                failed=progress.failed_documents,
            )

        return progress

    def get_migration_progress(
        self, source_version: str, target_version: str
    ) -> MigrationProgress | None:
        """Get migration progress.

        Args:
            source_version: Source version ID.
            target_version: Target version ID.

        Returns:
            Migration progress or None.
        """
        migration_id = f"{source_version}_to_{target_version}"
        return self._migrations.get(migration_id)

    def get_collection_for_search(self) -> list[str]:
        """Get collection names to search.

        During migration, returns both active and migrating collections
        to ensure no data is missed.

        Returns:
            List of collection names to search.
        """
        collections = []

        for version in self._versions.values():
            if version.status in (
                EmbeddingVersionStatus.ACTIVE,
                EmbeddingVersionStatus.MIGRATING,
            ):
                collections.append(version.collection_name)

        return collections


# Singleton instance
_version_manager: EmbeddingVersionManager | None = None


def get_version_manager() -> EmbeddingVersionManager:
    """Get or create version manager instance.

    Returns:
        Version manager singleton.
    """
    global _version_manager
    if _version_manager is None:
        _version_manager = EmbeddingVersionManager()
    return _version_manager
