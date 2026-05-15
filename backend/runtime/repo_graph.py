from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class RepositoryGraph:
    dependencies: dict[str, set[str]] = field(default_factory=dict)

    def add_dependency(self, *, source: str, target: str) -> None:
        self.dependencies.setdefault(source, set()).add(target)

    def neighbors(self, module: str) -> set[str]:
        return self.dependencies.get(module, set())

    def all_modules(self) -> set[str]:
        """All modules known to the graph — sources and targets."""
        modules: set[str] = set(self.dependencies.keys())
        for targets in self.dependencies.values():
            modules.update(targets)
        return modules

    def reverse(self) -> RepositoryGraph:
        """Return a new graph with all edges inverted — useful for reverse impact analysis."""
        inverted = RepositoryGraph()
        for source, targets in self.dependencies.items():
            for target in targets:
                inverted.add_dependency(source=target, target=source)
        return inverted

    @property
    def is_empty(self) -> bool:
        return not self.dependencies