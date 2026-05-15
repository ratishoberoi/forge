from __future__ import annotations
from dataclasses import dataclass
from backend.runtime.convergence_metrics import ConvergenceMetrics


@dataclass(slots=True)
class GlobalConvergenceState:
    iteration: int
    metrics: ConvergenceMetrics

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}.")