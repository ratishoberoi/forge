from __future__ import annotations
from backend.runtime.artifact_revision import ArtifactRevision
from backend.runtime.revision_graph import RevisionGraph


class RevisionTrackerError(Exception):
    """Raised when RevisionTracker encounters an invalid operation."""


class RevisionTracker:
    """
    High-level revision tracking over a RevisionGraph.
    Responsibilities:
    - track revisions into graph
    - traverse descendants and ancestors
    - find latest revision per artifact
    - list revisions by role
    - detect branching
    """

    def __init__(self, graph: RevisionGraph) -> None:
        self.graph = graph

    def track(self, revision: ArtifactRevision) -> None:
        """Add a revision to the graph."""
        if self.graph.contains(revision.revision_id):
            raise RevisionTrackerError(
                f"Revision '{revision.revision_id}' already tracked."
            )
        self.graph.add_revision(revision)

    def descendants(self, revision_id: str) -> set[str]:
        """Return all descendant revision_ids via BFS."""
        if not self.graph.contains(revision_id):
            raise RevisionTrackerError(
                f"Revision '{revision_id}' not found in graph."
            )
        visited: set[str] = set()
        stack = [revision_id]
        while stack:
            current = stack.pop()
            for child in self.graph.children(current):
                if child in visited:
                    continue
                visited.add(child)
                stack.append(child)
        return visited

    def ancestors(self, revision_id: str) -> set[str]:
        """Return all ancestor revision_ids."""
        if not self.graph.contains(revision_id):
            raise RevisionTrackerError(
                f"Revision '{revision_id}' not found in graph."
            )
        return self.graph.ancestors(revision_id)

    def lineage(self, revision_id: str) -> list[ArtifactRevision]:
        """Return ordered lineage from root to revision_id."""
        if not self.graph.contains(revision_id):
            raise RevisionTrackerError(
                f"Revision '{revision_id}' not found in graph."
            )
        return self.graph.lineage(revision_id)

    def latest_for_artifact(self, artifact_id: str) -> ArtifactRevision | None:
        """Return the most recently created revision for an artifact_id."""
        candidates = [
            r for r in self.graph.revisions.values()
            if r.artifact_id == artifact_id
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.created_at)

    def revisions_by_role(self, role: str) -> list[ArtifactRevision]:
        """Return all revisions contributed by a specific role, oldest first."""
        return sorted(
            [r for r in self.graph.revisions.values() if r.role.lower() == role.lower()],
            key=lambda r: r.created_at,
        )

    def is_branched(self, revision_id: str) -> bool:
        """True if a revision has more than one child (branch point)."""
        return len(self.graph.children(revision_id)) > 1

    @property
    def revision_count(self) -> int:
        return len(self.graph)

    @property
    def roots(self) -> list[ArtifactRevision]:
        return self.graph.roots()

    @property
    def leaves(self) -> list[ArtifactRevision]:
        return self.graph.leaves()