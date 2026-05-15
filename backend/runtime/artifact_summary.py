from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass(slots=True)
class ArtifactSummary:
    """
    Condensed representation of a multi-role cognition session.
    Produced after merging and summarizing CognitionArtifacts.
    Used for:
    - LLM context injection
    - audit trails
    - cross-round replay
    """
    summary_id: str
    source_roles: list[str]
    rounds: list[int]
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.summary_id:
            raise ValueError("summary_id must not be empty.")
        if not self.source_roles:
            raise ValueError("source_roles must not be empty.")
        if not self.rounds:
            raise ValueError("rounds must not be empty.")
        if not self.content.strip():
            raise ValueError("content must not be blank.")

    @property
    def round_span(self) -> tuple[int, int]:
        """Return (min_round, max_round) covered by this summary."""
        return min(self.rounds), max(self.rounds)

    @property
    def role_count(self) -> int:
        """Number of distinct roles contributing to this summary."""
        return len(set(self.source_roles))

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since summary was created."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def covers_round(self, round_id: int) -> bool:
        """Return True if round_id is included in this summary."""
        return round_id in self.rounds

    def covers_role(self, role: str) -> bool:
        """Return True if role contributed to this summary (case-insensitive)."""
        return role.lower() in {r.lower() for r in self.source_roles}

    def with_metadata(self, **kwargs: Any) -> ArtifactSummary:
        """Return a copy with additional metadata merged in."""
        return ArtifactSummary(
            summary_id=self.summary_id,
            source_roles=self.source_roles,
            rounds=self.rounds,
            content=self.content,
            created_at=self.created_at,
            metadata={**self.metadata, **kwargs},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict — for logging, storage, or API response."""
        return {
            "summary_id": self.summary_id,
            "source_roles": self.source_roles,
            "rounds": self.rounds,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        source_roles: list[str],
        rounds: list[int],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactSummary:
        """
        Factory method — auto-generates summary_id and timestamp.
        Preferred over direct construction in production code.
        """
        return cls(
            summary_id=str(uuid.uuid4()),
            source_roles=source_roles,
            rounds=rounds,
            content=content,
            metadata=metadata or {},
        )