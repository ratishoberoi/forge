from __future__ import annotations
from backend.runtime.repo_graph import RepositoryGraph


class RepositoryImpactAnalyzer:
    """
    Repository-aware impact analysis.
    Responsibilities:
    - transitive dependency traversal
    - blast radius estimation
    - reverse impact (who depends on X)
    """

    def impacted_modules(
        self,
        *,
        graph: RepositoryGraph,
        module: str,
    ) -> set[str]:
        """Forward traversal — all modules reachable from source."""
        visited: set[str] = set()
        stack = [module]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(graph.neighbors(current))
        return visited

    def blast_radius(
        self,
        *,
        graph: RepositoryGraph,
        module: str,
    ) -> int:
        """Number of modules transitively impacted by mutating this module."""
        return len(self.impacted_modules(graph=graph, module=module))

    def reverse_impact(
        self,
        *,
        graph: RepositoryGraph,
        module: str,
    ) -> set[str]:
        """Which modules depend on this module — reverse traversal."""
        return self.impacted_modules(graph=graph.reverse(), module=module)