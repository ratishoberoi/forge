from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class GraphExecutionState:
    completed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)

    def mark_complete(self, node_id: str) -> None:
        self.failed.discard(node_id)
        self.completed.add(node_id)

    def mark_failed(self, node_id: str) -> None:
        self.completed.discard(node_id)
        self.failed.add(node_id)

    @property
    def is_clean(self) -> bool:
        return not self.failed

    @property
    def total_completed(self) -> int:
        return len(self.completed)