from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass(slots=True)
class CognitionArtifact:
    """
    Immutable record of a single cognition output.
    Produced by any runtime role (coder, judge, embedder).
    Stored per-round for tracing, replay, and audit.
    """
    artifact_id: str
    role: str
    round_id: int
    task: str
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must not be empty.")
        if not self.role:
            raise ValueError("role must not be empty.")
        if not self.task:
            raise ValueError("task must not be empty.")
        if self.round_id < 0:
            raise ValueError(f"round_id must be >= 0, got {self.round_id}.")

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since artifact was created."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    @property
    def is_empty(self) -> bool:
        """True if content is blank or whitespace only."""
        return not self.content.strip()

    def with_metadata(self, **kwargs: Any) -> CognitionArtifact:
        """Return a copy with additional metadata merged in."""
        return CognitionArtifact(
            artifact_id=self.artifact_id,
            role=self.role,
            round_id=self.round_id,
            task=self.task,
            content=self.content,
            created_at=self.created_at,
            metadata={**self.metadata, **kwargs},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict — for logging, storage, or API response."""
        return {
            "artifact_id": self.artifact_id,
            "role": self.role,
            "round_id": self.round_id,
            "task": self.task,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        role: str,
        round_id: int,
        task: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> CognitionArtifact:
        """
        Factory method — auto-generates artifact_id and timestamp.
        Preferred over direct construction in production code.
        """
        return cls(
            artifact_id=str(uuid.uuid4()),
            role=role,
            round_id=round_id,
            task=task,
            content=content,
            metadata=metadata or {},
        )