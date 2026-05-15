from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass(slots=True)
class CourtroomArtifact:
    """
    Shared cognition artifact passed between courtroom roles.
    Evolves across rounds as critiques and revisions accumulate.
    """
    artifact_id: str
    objective: str
    patch: str
    critiques: list[str] = field(default_factory=list)
    revisions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must not be empty.")
        if not self.objective:
            raise ValueError("objective must not be empty.")
        if not self.patch:
            raise ValueError("patch must not be empty.")

    @property
    def revision_count(self) -> int:
        return len(self.revisions)

    @property
    def critique_count(self) -> int:
        return len(self.critiques)

    @property
    def has_critiques(self) -> bool:
        return bool(self.critiques)

    @property
    def latest_revision(self) -> str | None:
        return self.revisions[-1] if self.revisions else None

    def add_critique(self, critique: str) -> None:
        if not critique.strip():
            raise ValueError("Critique must not be blank.")
        self.critiques.append(critique)

    def add_revision(self, revision: str) -> None:
        if not revision.strip():
            raise ValueError("Revision must not be blank.")
        self.revisions.append(revision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "objective": self.objective,
            "patch": self.patch,
            "critiques": self.critiques,
            "revisions": self.revisions,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        objective: str,
        patch: str,
        metadata: dict[str, Any] | None = None,
    ) -> CourtroomArtifact:
        """Factory — auto-generates artifact_id and timestamp."""
        return cls(
            artifact_id=str(uuid.uuid4()),
            objective=objective,
            patch=patch,
            metadata=metadata or {},
        )