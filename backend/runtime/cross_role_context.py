from __future__ import annotations
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_merge import ArtifactMerger
from backend.runtime.artifact_query import ArtifactQueryEngine


class CrossRoleContextBuilder:
    """
    Builds unified cross-role context strings for LLM prompts.
    Responsibilities:
    - load artifacts across multiple roles
    - filter by round, metadata, or content
    - merge into LLM-readable context blocks
    - support targeted context windows
    """

    def __init__(
        self,
        *,
        query_engine: ArtifactQueryEngine,
        merger: ArtifactMerger,
    ) -> None:
        self.query_engine = query_engine
        self.merger = merger

    # ── Core build ────────────────────────────────────────────────────────────

    def build(self, roles: list[str]) -> str:
        """Load and merge all artifacts for given roles."""
        artifacts = self.query_engine.load_multi_role_artifacts(roles)
        return self.merger.merge(artifacts)

    def build_for_round(
        self,
        roles: list[str],
        round_id: int,
    ) -> str:
        """Load and merge artifacts for given roles filtered to a specific round."""
        artifacts = self.query_engine.load_multi_role_artifacts(roles)
        filtered = self.query_engine.filter_by_round(artifacts, round_id)
        return self.merger.merge(filtered)

    def build_for_round_range(
        self,
        roles: list[str],
        *,
        start: int,
        end: int,
    ) -> str:
        """Load and merge artifacts for given roles within a round range."""
        artifacts = self.query_engine.load_multi_role_artifacts(roles)
        filtered = self.query_engine.filter_by_round_range(artifacts, start=start, end=end)
        return self.merger.merge(filtered)

    def build_latest(
        self,
        roles: list[str],
        *,
        n_rounds: int = 3,
    ) -> str:
        """
        Build context from the n most recent rounds across all given roles.
        Useful for sliding-window prompts in long autonomous runs.
        """
        artifacts = self.query_engine.load_multi_role_artifacts(roles)
        if not artifacts:
            return ""

        all_rounds = self.query_engine.round_ids(artifacts)
        recent_rounds = all_rounds[-n_rounds:]
        filtered = self.query_engine.filter_by_round_range(
            artifacts,
            start=recent_rounds[0],
            end=recent_rounds[-1],
        )
        return self.merger.merge(filtered)

    def build_by_metadata(
        self,
        roles: list[str],
        **filters: object,
    ) -> str:
        """Load and merge artifacts filtered by metadata key=value pairs."""
        artifacts = self.query_engine.load_multi_role_artifacts(roles)
        filtered = self.query_engine.filter_by_metadata(artifacts, **filters)
        return self.merger.merge(filtered)

    def build_by_content(
        self,
        roles: list[str],
        substring: str,
        *,
        case_sensitive: bool = False,
    ) -> str:
        """Load and merge artifacts whose content contains the given substring."""
        artifacts = self.query_engine.load_multi_role_artifacts(roles)
        filtered = self.query_engine.filter_by_content(
            artifacts,
            substring,
            case_sensitive=case_sensitive,
        )
        return self.merger.merge(filtered)

    def latest_per_role(self, roles: list[str]) -> dict[str, CognitionArtifact]:
        """Return the latest artifact per role — for final state inspection."""
        artifacts = self.query_engine.load_multi_role_artifacts(roles)
        return self.query_engine.latest_per_role(artifacts)

    def roles_present(self, roles: list[str]) -> list[str]:
        """Return which of the given roles actually have artifacts on disk."""
        artifacts = self.query_engine.load_multi_role_artifacts(roles)
        return self.query_engine.roles_present(artifacts)