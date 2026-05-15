from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class PlanningState:
    round_count: int
    execution_failures: int
    stagnation_count: int
    architecture_risk: float
    judge_confidence: float
    context_pressure: float