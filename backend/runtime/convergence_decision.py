from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class ConvergenceDecision:
    converged: bool = False
    escalate: bool = False
    terminate: bool = False
    reason: str = ""