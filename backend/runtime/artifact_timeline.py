from __future__ import annotations
from dataclasses import dataclass, field
from backend.runtime.artifact import CognitionArtifact


@dataclass(slots=True)
class ArtifactTimeline:
    """
    Ordered view over a collection of CognitionArtifacts.
    Responsibilities:
    - deterministic ordering by (round_id, role)
    - round and role filtering
    - timeline slicing and windowing
    - per-round and per-role grouping
    - basic statistics
    """
    artifacts: list[CognitionArtifact] = field(default_factory=list)

    # ── Ordering ──────────────────────────────────────────────────────────────

    def ordered(self) -> list[CognitionArtifact]:
        """Return all artifacts sorted by (round_id, role)."""
        return sorted(self.artifacts, key=lambda a: (a.round_id, a.role))

    # ── Filters ───────────────────────────────────────────────────────────────

    def for_role(self, role: str) -> list[CognitionArtifact]:
        """Return artifacts for a specific role (case-insensitive), ordered."""
        return [
            a for a in self.ordered()
            if a.role.lower() == role.lower()
        ]

    def for_round(self, round_id: int) -> list[CognitionArtifact]:
        """Return artifacts for a specific round_id, ordered."""
        return [a for a in self.ordered() if a.round_id == round_id]

    def for_round_range(self, *, start: int, end: int) -> list[CognitionArtifact]:
        """Return artifacts within [start, end] round_id range inclusive, ordered."""
        return [a for a in self.ordered() if start <= a.round_id <= end]

    def latest(self, *, n: int = 3) -> list[CognitionArtifact]:
        """Return the n most recent artifacts by round_id."""
        return self.ordered()[-n:]

    def earliest(self, *, n: int = 3) -> list[CognitionArtifact]:
        """Return the n earliest artifacts by round_id."""
        return self.ordered()[:n]

    # ── Grouping ──────────────────────────────────────────────────────────────

    def grouped_by_round(self) -> dict[int, list[CognitionArtifact]]:
        """Return artifacts grouped by round_id, sorted within each group."""
        groups: dict[int, list[CognitionArtifact]] = {}
        for artifact in self.ordered():
            groups.setdefault(artifact.round_id, []).append(artifact)
        return groups

    def grouped_by_role(self) -> dict[str, list[CognitionArtifact]]:
        """Return artifacts grouped by role, sorted within each group."""
        groups: dict[str, list[CognitionArtifact]] = {}
        for artifact in self.ordered():
            groups.setdefault(artifact.role, []).append(artifact)
        return groups

    def latest_per_role(self) -> dict[str, CognitionArtifact]:
        """Return the highest round_id artifact per role."""
        result: dict[str, CognitionArtifact] = {}
        for artifact in self.ordered():
            result[artifact.role] = artifact
        return result

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def round_ids(self) -> list[int]:
        """Sorted unique round_ids present in timeline."""
        return sorted({a.round_id for a in self.artifacts})

    @property
    def roles(self) -> list[str]:
        """Sorted unique roles present in timeline."""
        return sorted({a.role for a in self.artifacts})

    @property
    def round_count(self) -> int:
        """Number of distinct rounds in timeline."""
        return len(self.round_ids)

    @property
    def is_empty(self) -> bool:
        """True if timeline contains no artifacts."""
        return not self.artifacts

    def __len__(self) -> int:
        return len(self.artifacts)

    def __contains__(self, artifact_id: str) -> bool:
        """Support `artifact_id in timeline` membership check."""
        return any(a.artifact_id == artifact_id for a in self.artifacts)