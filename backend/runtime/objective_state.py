from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class ObjectiveState:
    completed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return bool(self.pending == [] and not self.failed)

    @property
    def has_failures(self) -> bool:
        return bool(self.failed)

    @property
    def progress_ratio(self) -> float:
        total = len(self.completed) + len(self.pending) + len(self.failed)
        if total == 0:
            return 0.0
        return len(self.completed) / total

    def mark_complete(self, subgoal: str) -> None:
        if subgoal in self.pending:
            self.pending.remove(subgoal)
        if subgoal not in self.completed:
            self.completed.append(subgoal)

    def mark_failed(self, subgoal: str) -> None:
        if subgoal in self.pending:
            self.pending.remove(subgoal)
        if subgoal not in self.failed:
            self.failed.append(subgoal)