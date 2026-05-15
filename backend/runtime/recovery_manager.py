from __future__ import annotations
import uuid
from backend.runtime.recovery_checkpoint import RecoveryCheckpoint
from backend.runtime.recovery_policy import RecoveryPolicy
from backend.runtime.recovery_state import RecoveryState


class RecoveryManager:
    """
    Cognition fault tolerance manager.
    Responsibilities:
    - create recovery checkpoints
    - track recovery attempts
    - enforce recovery policy limits
    - determine checkpoint eligibility
    """

    def __init__(self, policy: RecoveryPolicy) -> None:
        self.policy = policy
        self._attempts: int = 0
        self._checkpoints: list[RecoveryCheckpoint] = []

    def create_checkpoint(self, state: RecoveryState) -> RecoveryCheckpoint:
        checkpoint = RecoveryCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            state=state,
        )
        self._checkpoints.append(checkpoint)
        return checkpoint

    def latest_checkpoint(self) -> RecoveryCheckpoint | None:
        return self._checkpoints[-1] if self._checkpoints else None

    def should_checkpoint(self, iteration: int) -> bool:
        """True if current iteration warrants a new checkpoint."""
        return iteration % self.policy.checkpoint_every_n_iterations == 0

    def can_recover(self) -> bool:
        return self._attempts < self.policy.max_recovery_attempts

    def mark_recovery_attempt(self) -> None:
        self._attempts += 1

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.policy.max_recovery_attempts - self._attempts)