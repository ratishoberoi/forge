from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class PlanningPolicy:
    max_rounds: int = 10
    retry_failure_threshold: int = 2
    stagnation_threshold: int = 3
    architecture_risk_threshold: float = 0.7
    confidence_threshold: float = 0.9
    context_pressure_threshold: float = 0.8