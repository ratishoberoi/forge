from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from backend.runtime.runtime_process import RuntimeProcess


@dataclass(slots=True)
class RuntimeSession:
    """
    Represents temporary ownership of a single active runtime.
    Tracks session lifecycle from boot to teardown.
    Responsibilities:
    - bind process to objective
    - record timing
    - carry per-session metadata
    """
    process: RuntimeProcess
    objective: str
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ended_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("objective must not be blank.")

    @property
    def is_active(self) -> bool:
        """True if session has not been ended."""
        return self.ended_at is None

    @property
    def duration_seconds(self) -> float | None:
        """Elapsed seconds from start to end. None if still active."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def role(self) -> str:
        """Convenience accessor for process role."""
        return self.process.role

    def end(self) -> None:
        """Mark session as ended."""
        if self.ended_at is not None:
            raise ValueError(
                f"Session for role '{self.role}' is already ended."
            )
        self.ended_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "objective": self.objective,
            "pid": self.process.pid,
            "port": self.process.port,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }