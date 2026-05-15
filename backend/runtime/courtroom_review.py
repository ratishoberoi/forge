from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

VALID_SEVERITIES = {"low", "medium", "high", "critical"}


@dataclass(slots=True)
class CourtroomReview:
    """
    Structured critique produced by a courtroom reviewer role.
    Attached to a CourtroomRound after evaluation.
    """
    reviewer_role: str
    critique: str
    severity: str
    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reviewer_role:
            raise ValueError("reviewer_role must not be empty.")
        if not self.critique.strip():
            raise ValueError("critique must not be blank.")
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {VALID_SEVERITIES}, got '{self.severity}'."
            )

    @property
    def is_blocking(self) -> bool:
        """High and critical severity reviews block acceptance."""
        return self.severity in {"high", "critical"}

    @property
    def is_critical(self) -> bool:
        return self.severity == "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "reviewer_role": self.reviewer_role,
            "critique": self.critique,
            "severity": self.severity,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        reviewer_role: str,
        critique: str,
        severity: str,
        metadata: dict[str, Any] | None = None,
    ) -> CourtroomReview:
        """Factory — auto-generates review_id and timestamp."""
        return cls(
            reviewer_role=reviewer_role,
            critique=critique,
            severity=severity,
            metadata=metadata or {},
        )