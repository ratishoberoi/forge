from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class PriorityPolicy:
    execution_weight: float = 0.4
    architecture_weight: float = 0.3
    retry_weight: float = 0.2
    instability_weight: float = 0.1

    def __post_init__(self) -> None:
        total = (
            self.execution_weight
            + self.architecture_weight
            + self.retry_weight
            + self.instability_weight
        )
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f}."
            )