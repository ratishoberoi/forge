from __future__ import annotations
from dataclasses import dataclass, field
from backend.runtime.artifact_revision import ArtifactRevision


@dataclass(slots=True)
class RevisionGraph:
    """
    Directed acyclic graph of ArtifactRevisions.
    Nodes: revision_id → ArtifactRevision
    Edges: parent_revision_id → set of child revision_ids
    Responsibilities:
    - track parent-child revision relationships
    - support ancestor and descendant traversal
    - detect root revisions
    - detect leaf revisions
    """
    revisions: dict[str, ArtifactRevision] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)

    def add_revision(self, revision: ArtifactRevision) -> None:
        """Add a revision node and wire its parent edge."""
        self.revisions[revision.revision_id] = revision
        self.edges.setdefault(revision.revision_id, set())

        if revision.parent_revision_id:
            self.edges.setdefault(
                revision.parent_revision_id, set()
            ).add(revision.revision_id)

    def children(self, revision_id: str) -> set[str]:
        """Return direct children of a revision."""
        return self.edges.get(revision_id, set())

    def parents(self, revision_id: str) -> set[str]:
        """Return direct parents of a revision (0 or 1 in a linear chain)."""
        revision = self.revisions.get(revision_id)
        if revision is None:
            return set()
        if revision.parent_revision_id is None:
            return set()
        return {revision.parent_revision_id}

    def roots(self) -> list[ArtifactRevision]:
        """Return all root revisions (no parent)."""
        return [r for r in self.revisions.values() if r.is_root]

    def leaves(self) -> list[ArtifactRevision]:
        """Return all leaf revisions (no children)."""
        return [
            r for r in self.revisions.values()
            if not self.edges.get(r.revision_id)
        ]

    def ancestors(self, revision_id: str) -> set[str]:
        """Return all ancestor revision_ids via parent chain traversal."""
        visited: set[str] = set()
        stack = list(self.parents(revision_id))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.parents(current))
        return visited

    def lineage(self, revision_id: str) -> list[ArtifactRevision]:
        """
        Return ordered lineage from root to this revision.
        Oldest first.
        """
        chain: list[str] = [revision_id]
        current = self.revisions.get(revision_id)
        while current and current.parent_revision_id:
            chain.append(current.parent_revision_id)
            current = self.revisions.get(current.parent_revision_id)
        chain.reverse()
        return [self.revisions[rid] for rid in chain if rid in self.revisions]

    def contains(self, revision_id: str) -> bool:
        return revision_id in self.revisions

    def __len__(self) -> int:
        return len(self.revisions)