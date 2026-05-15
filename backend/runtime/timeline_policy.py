from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class TimelinePolicy:
    """
    Controls how ArtifactTimeline is replayed and windowed for LLM context.
    Responsibilities:
    - cap total artifacts replayed
    - enforce recency windows
    - filter by role priority
    - control ordering strategy
    - enforce content size limits
    """
    max_replay_artifacts: int = 20
    max_content_chars: int = 64_000
    preserve_recent: int = 5
    priority_roles: list[str] = field(default_factory=list)
    drop_empty_content: bool = True
    deduplicate: bool = True
    order_by: str = "round_role"  # "round_role" | "round_only" | "role_round"

    def __post_init__(self) -> None:
        if self.max_replay_artifacts < 1:
            raise ValueError(
                f"max_replay_artifacts must be >= 1, got {self.max_replay_artifacts}."
            )
        if self.preserve_recent < 0:
            raise ValueError(
                f"preserve_recent must be >= 0, got {self.preserve_recent}."
            )
        if self.preserve_recent > self.max_replay_artifacts:
            raise ValueError(
                f"preserve_recent ({self.preserve_recent}) must be <= "
                f"max_replay_artifacts ({self.max_replay_artifacts})."
            )
        if self.max_content_chars < 1:
            raise ValueError(
                f"max_content_chars must be >= 1, got {self.max_content_chars}."
            )
        valid_orders = {"round_role", "round_only", "role_round"}
        if self.order_by not in valid_orders:
            raise ValueError(
                f"order_by must be one of {valid_orders}, got '{self.order_by}'."
            )

    @property
    def compressible_slots(self) -> int:
        """Artifact slots available beyond the preserved recent window."""
        return self.max_replay_artifacts - self.preserve_recent

    @property
    def has_priority_roles(self) -> bool:
        """True if specific roles should be surfaced before others."""
        return bool(self.priority_roles)

    def is_priority_role(self, role: str) -> bool:
        """Return True if role is in priority list (case-insensitive)."""
        return role.lower() in {r.lower() for r in self.priority_roles}

    def allows_content(self, content: str) -> bool:
        """Return True if content passes policy filters."""
        if self.drop_empty_content and not content.strip():
            return False
        return True

    def sort_key(self) -> object:
        """Return the sort key function matching order_by strategy."""
        from backend.runtime.artifact import CognitionArtifact

        strategies: dict[str, object] = {
            "round_role": lambda a: (a.round_id, a.role),
            "round_only": lambda a: a.round_id,
            "role_round": lambda a: (a.role, a.round_id),
        }
        return strategies[self.order_by]

    # ── Presets ───────────────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> TimelinePolicy:
        """Standard policy for most cognition runs."""
        return cls()

    @classmethod
    def tight(cls) -> TimelinePolicy:
        """Aggressive cap for long autonomous runs with large context pressure."""
        return cls(
            max_replay_artifacts=10,
            preserve_recent=3,
            max_content_chars=32_000,
            deduplicate=True,
            drop_empty_content=True,
        )

    @classmethod
    def judge_focused(cls) -> TimelinePolicy:
        """Surface JUDGE artifacts first — for coder refinement loops."""
        return cls(
            max_replay_artifacts=20,
            preserve_recent=5,
            priority_roles=["JUDGE"],
            order_by="role_round",
        )

    @classmethod
    def full_replay(cls) -> TimelinePolicy:
        """Unlimited replay — for audit, debugging, or post-run analysis."""
        return cls(
            max_replay_artifacts=1000,
            preserve_recent=0,
            max_content_chars=256_000,
            deduplicate=False,
            drop_empty_content=False,
        )