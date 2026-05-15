from __future__ import annotations
from dataclasses import dataclass

_VALID_REASONS = frozenset({"critical", "high", "normal"})


@dataclass(slots=True)
class PriorityScore:
    score: float
    reason: str

    @property
    def is_critical(self) -> bool:
        return self.reason == "critical"

    @property
    def is_high(self) -> bool:
        return self.reason == "high"

    @property
    def is_normal(self) -> bool:
        return self.reason == "normal"