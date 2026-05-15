from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class ConvergenceMetrics:
    score_variance: float
    retry_improvement: float
    critique_diversity: float
    execution_stability: float

    def __post_init__(self) -> None:
        for name, val in [
            ("score_variance", self.score_variance),
            ("retry_improvement", self.retry_improvement),
            ("critique_diversity", self.critique_diversity),
            ("execution_stability", self.execution_stability),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}.")