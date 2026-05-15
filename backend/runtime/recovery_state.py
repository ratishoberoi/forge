from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class RecoveryState:
    iteration: int
    active_role: str
    last_completed_node: str
    replay_offset: int

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}.")
        if self.replay_offset < 0:
            raise ValueError(f"replay_offset must be >= 0, got {self.replay_offset}.")
        if not self.active_role:
            raise ValueError("active_role must not be empty.")