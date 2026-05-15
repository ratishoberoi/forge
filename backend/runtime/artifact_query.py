from __future__ import annotations
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_loader import ArtifactLoader


class ArtifactQueryEngine:
    """
    High-level query interface over ArtifactLoader.
    Responsibilities:
    - multi-role artifact loading
    - filtering by round, role, metadata, content
    - deduplication
    - sorting strategies
    """

    def __init__(self, loader: ArtifactLoader) -> None:
        self.loader = loader

    # ── Multi-role load ───────────────────────────────────────────────────────

    def load_multi_role_artifacts(
        self,
        roles: list[str],
    ) -> list[CognitionArtifact]:
        """Load and merge artifacts for multiple roles, sorted by (round_id, role)."""
        artifacts: list[CognitionArtifact] = []
        for role in roles:
            artifacts.extend(self.loader.load_role_artifacts(role))
        return self._sort_by_round_role(artifacts)

    # ── Filters ───────────────────────────────────────────────────────────────

    def filter_by_round(
        self,
        artifacts: list[CognitionArtifact],
        round_id: int,
    ) -> list[CognitionArtifact]:
        """Return artifacts matching a specific round_id."""
        return [a for a in artifacts if a.round_id == round_id]

    def filter_by_role(
        self,
        artifacts: list[CognitionArtifact],
        role: str,
    ) -> list[CognitionArtifact]:
        """Return artifacts matching a specific role (case-insensitive)."""
        return [a for a in artifacts if a.role.lower() == role.lower()]

    def filter_by_round_range(
        self,
        artifacts: list[CognitionArtifact],
        *,
        start: int,
        end: int,
    ) -> list[CognitionArtifact]:
        """Return artifacts within [start, end] round_id range inclusive."""
        return [a for a in artifacts if start <= a.round_id <= end]

    def filter_by_metadata(
        self,
        artifacts: list[CognitionArtifact],
        **filters: object,
    ) -> list[CognitionArtifact]:
        """Return artifacts whose metadata contains all given key=value pairs."""
        return [
            a for a in artifacts
            if all(a.metadata.get(k) == v for k, v in filters.items())
        ]

    def filter_by_content(
        self,
        artifacts: list[CognitionArtifact],
        substring: str,
        *,
        case_sensitive: bool = False,
    ) -> list[CognitionArtifact]:
        """Return artifacts whose content contains the given substring."""
        needle = substring if case_sensitive else substring.lower()
        return [
            a for a in artifacts
            if needle in (a.content if case_sensitive else a.content.lower())
        ]

    # ── Aggregations ──────────────────────────────────────────────────────────

    def latest_per_role(
        self,
        artifacts: list[CognitionArtifact],
    ) -> dict[str, CognitionArtifact]:
        """Return the highest round_id artifact per role."""
        result: dict[str, CognitionArtifact] = {}
        for artifact in artifacts:
            existing = result.get(artifact.role)
            if existing is None or artifact.round_id > existing.round_id:
                result[artifact.role] = artifact
        return result

    def round_ids(self, artifacts: list[CognitionArtifact]) -> list[int]:
        """Return sorted unique round_ids present in artifact list."""
        return sorted({a.round_id for a in artifacts})

    def roles_present(self, artifacts: list[CognitionArtifact]) -> list[str]:
        """Return sorted unique roles present in artifact list."""
        return sorted({a.role for a in artifacts})

    def deduplicate(
        self,
        artifacts: list[CognitionArtifact],
    ) -> list[CognitionArtifact]:
        """
        Remove duplicate artifacts by artifact_id.
        Preserves first occurrence — maintains sort order.
        """
        seen: set[str] = set()
        result: list[CognitionArtifact] = []
        for artifact in artifacts:
            if artifact.artifact_id not in seen:
                seen.add(artifact.artifact_id)
                result.append(artifact)
        return result

    # ── Sorting ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sort_by_round_role(
        artifacts: list[CognitionArtifact],
    ) -> list[CognitionArtifact]:
        return sorted(artifacts, key=lambda a: (a.round_id, a.role))