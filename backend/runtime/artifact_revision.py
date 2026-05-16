from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass(slots=True)
class ArtifactRevision:
    """
    Single node in a cognition revision chain.
    Tracks parent linkage for full lineage replay.
    """
    revision_id: str
    artifact_id: str
    role: str
    summary: str
    parent_revision_id: str | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.revision_id:
            raise ValueError("revision_id must not be empty.")
        if not self.artifact_id:
            raise ValueError("artifact_id must not be empty.")
        if not self.role:
            raise ValueError("role must not be empty.")
        if not self.summary.strip():
            raise ValueError("summary must not be blank.")

    @property
    def is_root(self) -> bool:
        """True if this revision has no parent."""
        return self.parent_revision_id is None

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "artifact_id": self.artifact_id,
            "parent_revision_id": self.parent_revision_id,
            "role": self.role,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        role: str,
        summary: str,
        parent_revision_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRevision:
        """Factory — auto-generates revision_id and timestamp."""
        return cls(
            revision_id=str(uuid.uuid4()),
            artifact_id=artifact_id,
            role=role,
            summary=summary,
            parent_revision_id=parent_revision_id,
            metadata=metadata or {},
        )