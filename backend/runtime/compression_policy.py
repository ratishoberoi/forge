from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class CompressionPolicy:
    """
    Controls how artifact history is compressed before LLM context injection.
    Responsibilities:
    - cap total artifacts fed to context
    - preserve recent rounds unconditionally
    - filter by role priority
    - enforce content size limits
    """
    max_artifacts: int = 10
    preserve_recent: int = 3
    max_content_chars: int = 32_000
    priority_roles: list[str] = field(default_factory=list)
    drop_empty_content: bool = True
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if self.max_artifacts < 1:
            raise ValueError(f"max_artifacts must be >= 1, got {self.max_artifacts}.")
        if self.preserve_recent < 0:
            raise ValueError(f"preserve_recent must be >= 0, got {self.preserve_recent}.")
        if self.preserve_recent > self.max_artifacts:
            raise ValueError(
                f"preserve_recent ({self.preserve_recent}) must be <= "
                f"max_artifacts ({self.max_artifacts})."
            )
        if self.max_content_chars < 1:
            raise ValueError(
                f"max_content_chars must be >= 1, got {self.max_content_chars}."
            )

    @property
    def compressible_slots(self) -> int:
        """Number of artifact slots available beyond the preserved recent window."""
        return self.max_artifacts - self.preserve_recent

    @property
    def has_priority_roles(self) -> bool:
        """True if specific roles should be prioritized during compression."""
        return bool(self.priority_roles)

    def is_priority_role(self, role: str) -> bool:
        """Return True if role is in the priority list (case-insensitive)."""
        return role.lower() in {r.lower() for r in self.priority_roles}

    def allows_content(self, content: str) -> bool:
        """
        Return True if content passes policy filters.
        Rejects blank content when drop_empty_content is set.
        """
        if self.drop_empty_content and not content.strip():
            return False
        return True

    @classmethod
    def default(cls) -> CompressionPolicy:
        """Standard policy for most cognition runs."""
        return cls()

    @classmethod
    def aggressive(cls) -> CompressionPolicy:
        """Tight policy for long autonomous runs with large context pressure."""
        return cls(
            max_artifacts=5,
            preserve_recent=2,
            max_content_chars=16_000,
            deduplicate=True,
            drop_empty_content=True,
        )

    @classmethod
    def judge_focused(cls) -> CompressionPolicy:
        """Prioritize JUDGE artifacts — for coder refinement loops."""
        return cls(
            max_artifacts=10,
            preserve_recent=3,
            priority_roles=["JUDGE"],
            deduplicate=True,
        )