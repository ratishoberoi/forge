from __future__ import annotations
from backend.runtime.artifact import CognitionArtifact


class ArtifactMerger:
    """
    Merges CognitionArtifacts into unified text representations.
    Responsibilities:
    - merge all artifacts into a single string
    - merge by role into separate sections
    - merge latest-per-round only
    - support custom separators and headers
    """

    DEFAULT_SEPARATOR = "\n\n"

    # ── Core merge ────────────────────────────────────────────────────────────

    def merge(
        self,
        artifacts: list[CognitionArtifact],
        *,
        separator: str = DEFAULT_SEPARATOR,
    ) -> str:
        """Merge all artifacts into a single string, sorted by (round_id, role)."""
        if not artifacts:
            return ""

        sorted_artifacts = sorted(artifacts, key=lambda a: (a.round_id, a.role))
        sections = [self._format(a) for a in sorted_artifacts]
        return separator.join(sections)

    def merge_by_role(
        self,
        artifacts: list[CognitionArtifact],
        *,
        separator: str = DEFAULT_SEPARATOR,
    ) -> dict[str, str]:
        """
        Merge artifacts grouped by role.
        Returns dict of role -> merged content string.
        """
        grouped: dict[str, list[CognitionArtifact]] = {}
        for artifact in artifacts:
            grouped.setdefault(artifact.role, []).append(artifact)

        return {
            role: self.merge(role_artifacts, separator=separator)
            for role, role_artifacts in grouped.items()
        }

    def merge_latest_per_round(
        self,
        artifacts: list[CognitionArtifact],
        *,
        separator: str = DEFAULT_SEPARATOR,
    ) -> str:
        """
        Merge only the latest artifact per round_id (by role alphabetically last).
        Useful when multiple roles contribute to the same round.
        """
        latest: dict[int, CognitionArtifact] = {}
        for artifact in artifacts:
            existing = latest.get(artifact.round_id)
            if existing is None or artifact.role > existing.role:
                latest[artifact.round_id] = artifact

        return self.merge(list(latest.values()), separator=separator)

    def merge_roles(
        self,
        artifacts: list[CognitionArtifact],
        roles: list[str],
        *,
        separator: str = DEFAULT_SEPARATOR,
    ) -> str:
        """Merge artifacts filtered to a specific set of roles."""
        role_set = {r.lower() for r in roles}
        filtered = [a for a in artifacts if a.role.lower() in role_set]
        return self.merge(filtered, separator=separator)

    def merge_round_range(
        self,
        artifacts: list[CognitionArtifact],
        *,
        start: int,
        end: int,
        separator: str = DEFAULT_SEPARATOR,
    ) -> str:
        """Merge artifacts within [start, end] round_id range inclusive."""
        filtered = [a for a in artifacts if start <= a.round_id <= end]
        return self.merge(filtered, separator=separator)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _format(artifact: CognitionArtifact) -> str:
        return (
            f"=== ROLE {artifact.role} ROUND {artifact.round_id} "
            f"| {artifact.task} ===\n"
            f"{artifact.content}"
        )