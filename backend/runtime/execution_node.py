from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class ExecutionNode:
    node_id: str
    title: str
    dependencies: list[str] = field(default_factory=list)

    @property
    def has_dependencies(self) -> bool:
        return bool(self.dependencies)