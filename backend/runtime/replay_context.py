from __future__ import annotations
from backend.runtime.artifact import CognitionArtifact


class ReplayContextBuilder:
    """
    Builds structured replay context strings from cognition artifacts.
    Used to inject prior round outputs into subsequent prompts.
    """

    def build(self, artifacts: list[CognitionArtifact]) -> str:
        """Build ordered replay context from artifacts."""
        if not artifacts:
            return ""

        sorted_artifacts = sorted(artifacts, key=lambda a: a.round_id)

        sections: list[str] = []
        for artifact in sorted_artifacts:
            if artifact.is_empty:
                continue
            sections.append(
                f"[REPLAY ROLE={artifact.role} ROUND={artifact.round_id}]\n"
                f"{artifact.content}"
            )

        return "\n\n".join(sections)

    def build_for_role(
        self,
        artifacts: list[CognitionArtifact],
        role: str,
    ) -> str:
        """Build replay context filtered to a specific role."""
        return self.build([a for a in artifacts if a.role == role])

    def build_latest_round(
        self,
        artifacts: list[CognitionArtifact],
    ) -> str:
        """Build replay context from the latest round only."""
        if not artifacts:
            return ""
        latest = max(a.round_id for a in artifacts)
        return self.build([a for a in artifacts if a.round_id == latest])