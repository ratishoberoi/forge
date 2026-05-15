from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class PrioritySignal:
    execution_risk: float
    architecture_risk: float
    retry_pressure: float
    convergence_instability: float

    def __post_init__(self) -> None:
        for name, val in [
            ("execution_risk", self.execution_risk),
            ("architecture_risk", self.architecture_risk),
            ("retry_pressure", self.retry_pressure),
            ("convergence_instability", self.convergence_instability),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}.")