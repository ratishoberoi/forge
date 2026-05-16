from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class RuntimeProcess:
    """
    Metadata for a single runtime inference process.
    Tracks lifecycle state from spawn to shutdown.
    """
    role: str
    model: str
    port: int
    pid: int | None = None
    active: bool = False
    launched_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("role must not be empty.")
        if not self.model:
            raise ValueError("model must not be empty.")
        if not (1024 <= self.port <= 65535):
            raise ValueError(
                f"port must be between 1024 and 65535, got {self.port}."
            )

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL for this process."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def is_running(self) -> bool:
        """True if process has a PID and is marked active."""
        return self.pid is not None and self.active

    @property
    def uptime_seconds(self) -> float | None:
        """Seconds since launch, or None if not yet launched."""
        if self.launched_at is None:
            return None
        return (datetime.now(timezone.utc) - self.launched_at).total_seconds()

    def mark_launched(self, pid: int) -> None:
        """Set PID, mark active, record launch time."""
        self.pid = pid
        self.active = True
        self.launched_at = datetime.now(timezone.utc)

    def mark_stopped(self) -> None:
        """Mark process as inactive."""
        self.active = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.model,
            "port": self.port,
            "pid": self.pid,
            "active": self.active,
            "launched_at": self.launched_at.isoformat() if self.launched_at else None,
            "metadata": self.metadata,
        }