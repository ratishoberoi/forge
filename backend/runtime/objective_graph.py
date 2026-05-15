from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class ObjectiveGraph:
    dependencies: dict[str, list[str]] = field(default_factory=dict)

    def add_dependency(self, *, subgoal: str, depends_on: str) -> None:
        self.dependencies.setdefault(subgoal, []).append(depends_on)

    def dependencies_for(self, subgoal: str) -> list[str]:
        return self.dependencies.get(subgoal, [])

    def has_dependencies(self, subgoal: str) -> bool:
        return bool(self.dependencies_for(subgoal))

    def ready_subgoals(
        self,
        all_subgoals: list[str],
        completed: list[str],
    ) -> list[str]:
        """Return subgoals whose dependencies are all completed."""
        completed_set = set(completed)
        return [
            s for s in all_subgoals
            if all(dep in completed_set for dep in self.dependencies_for(s))
            and s not in completed_set
        ]