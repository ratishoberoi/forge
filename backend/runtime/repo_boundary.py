from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class RepositoryBoundaryPolicy:
    """
    Architectural boundary enforcement.
    Prevents autonomous mutations from touching protected modules.
    """
    protected_modules: set[str] = field(default_factory=set)

    def allows(self, module: str) -> bool:
        return module not in self.protected_modules

    def protect(self, *modules: str) -> None:
        self.protected_modules.update(modules)

    def violation_reason(self, module: str) -> str | None:
        if not self.allows(module):
            return f"Module '{module}' is protected — mutation not allowed."
        return None