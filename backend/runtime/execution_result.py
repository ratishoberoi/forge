from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ExecutionResult:
    """
    Captures the full output of a subprocess execution.
    Immutable record — produced by ExecutionRunner.
    """
    command: list[str]
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0
    executed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0

    @property
    def failed(self) -> bool:
        return not self.succeeded

    @property
    def command_str(self) -> str:
        """Human-readable command string."""
        return " ".join(self.command)

    @property
    def has_stdout(self) -> bool:
        return bool(self.stdout.strip())

    @property
    def has_stderr(self) -> bool:
        return bool(self.stderr.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "succeeded": self.succeeded,
            "duration_seconds": self.duration_seconds,
            "executed_at": self.executed_at.isoformat(),
            "metadata": self.metadata,
        }