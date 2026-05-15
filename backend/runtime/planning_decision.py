from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class PlanningDecision:
    invoke_retry: bool = False
    invoke_architect: bool = False
    invoke_judge: bool = True
    compress_context: bool = False
    terminate: bool = False
    reason: str = ""