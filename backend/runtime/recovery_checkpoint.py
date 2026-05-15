from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from backend.runtime.recovery_state import RecoveryState


@dataclass(slots=True)
class RecoveryCheckpoint:
    checkpoint_id: str
    state: RecoveryState
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()