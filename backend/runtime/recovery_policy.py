from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class RecoveryPolicy:
    max_recovery_attempts: int = 3
    checkpoint_every_n_iterations: int = 1
    rollback_on_failure: bool = True

    def __post_init__(self) -> None:
        if self.max_recovery_attempts < 1:
            raise ValueError(
                f"max_recovery_attempts must be >= 1, got {self.max_recovery_attempts}."
            )
        if self.checkpoint_every_n_iterations < 1:
            raise ValueError(
                f"checkpoint_every_n_iterations must be >= 1, "
                f"got {self.checkpoint_every_n_iterations}."
            )