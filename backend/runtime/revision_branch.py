from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class RevisionBranch:
    """
    Tracks a named sequence of revision_ids forming a branch.
    Responsibilities:
    - maintain ordered revision sequence
    - track branch origin (root revision)
    - report tip (latest revision)
    - support branch length and membership queries
    """
    branch_id: str
    revisions: list[str] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.branch_id:
            raise ValueError("branch_id must not be empty.")

    def append(self, revision_id: str) -> None:
        """Add a revision to the tip of this branch."""
        if not revision_id:
            raise ValueError("revision_id must not be empty.")
        if revision_id in self.revisions:
            raise ValueError(
                f"Revision '{revision_id}' already exists in branch '{self.branch_id}'."
            )
        self.revisions.append(revision_id)

    @property
    def tip(self) -> str | None:
        """Most recent revision in this branch."""
        return self.revisions[-1] if self.revisions else None

    @property
    def root(self) -> str | None:
        """First revision in this branch."""
        return self.revisions[0] if self.revisions else None

    @property
    def length(self) -> int:
        return len(self.revisions)

    @property
    def is_empty(self) -> bool:
        return not self.revisions

    def contains(self, revision_id: str) -> bool:
        return revision_id in self.revisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "revisions": self.revisions,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }