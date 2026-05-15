from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class Objective:
    objective_id: str
    title: str
    description: str
    subgoals: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.subgoals

    @property
    def subgoal_count(self) -> int:
        return len(self.subgoals)