from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid
from backend.runtime.courtroom_artifact import CourtroomArtifact
from backend.runtime.courtroom_review import CourtroomReview


@dataclass(slots=True)
class CourtroomRound:
    """
    Encapsulates one full courtroom deliberation round.
    Contains the artifact under review and all reviews produced.
    """
    round_id: str
    artifact: CourtroomArtifact
    reviews: list[CourtroomReview] = field(default_factory=list)
    accepted: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.round_id:
            raise ValueError("round_id must not be empty.")

    @property
    def review_count(self) -> int:
        return len(self.reviews)

    @property
    def has_blocking_reviews(self) -> bool:
        """True if any review has high or critical severity."""
        return any(r.is_blocking for r in self.reviews)

    @property
    def has_critical_reviews(self) -> bool:
        """True if any review has critical severity."""
        return any(r.is_critical for r in self.reviews)

    @property
    def blocking_reviews(self) -> list[CourtroomReview]:
        return [r for r in self.reviews if r.is_blocking]

    @property
    def reviews_by_role(self) -> dict[str, list[CourtroomReview]]:
        """Group reviews by reviewer_role."""
        groups: dict[str, list[CourtroomReview]] = {}
        for review in self.reviews:
            groups.setdefault(review.reviewer_role, []).append(review)
        return groups

    @property
    def severity_summary(self) -> dict[str, int]:
        """Count reviews per severity level."""
        summary: dict[str, int] = {}
        for review in self.reviews:
            summary[review.severity] = summary.get(review.severity, 0) + 1
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "artifact": self.artifact.to_dict(),
            "reviews": [r.to_dict() for r in self.reviews],
            "accepted": self.accepted,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        artifact: CourtroomArtifact,
        metadata: dict[str, Any] | None = None,
    ) -> CourtroomRound:
        """Factory — auto-generates round_id and timestamp."""
        return cls(
            round_id=str(uuid.uuid4()),
            artifact=artifact,
            metadata=metadata or {},
        )