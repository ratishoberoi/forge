from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass(slots=True)
class RevisionMerge:
    """
    Records a merge event between two revision branches.
    source_revision: the branch being merged in
    target_revision: the branch being merged into
    merged_revision: the resulting merged revision_id
    """
    source_revision: str
    target_revision: str
    merged_revision: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_revision:
            raise ValueError("source_revision must not be empty.")
        if not self.target_revision:
            raise ValueError("target_revision must not be empty.")
        if not self.merged_revision:
            raise ValueError("merged_revision must not be empty.")
        if self.source_revision == self.target_revision:
            raise ValueError(
                "source_revision and target_revision must differ."
            )

    @property
    def involves(self) -> set[str]:
        """All revision_ids involved in this merge."""
        return {self.source_revision, self.target_revision, self.merged_revision}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_revision": self.source_revision,
            "target_revision": self.target_revision,
            "merged_revision": self.merged_revision,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        source_revision: str,
        target_revision: str,
        metadata: dict[str, Any] | None = None,
    ) -> RevisionMerge:
        """Factory — auto-generates merged_revision id."""
        return cls(
            source_revision=source_revision,
            target_revision=target_revision,
            merged_revision=str(uuid.uuid4()),
            metadata=metadata or {},
        )