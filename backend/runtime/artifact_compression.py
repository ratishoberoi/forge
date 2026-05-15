from __future__ import annotations
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_summary import ArtifactSummary
from backend.runtime.compression_policy import CompressionPolicy


class ArtifactCompressor:
    """
    Compresses CognitionArtifact history into ArtifactSummary.
    Responsibilities:
    - decide when compression is needed
    - split artifacts into compressible vs preserved window
    - apply policy filters (empty content, deduplication, priority roles)
    - produce ArtifactSummary via factory method
    """

    def __init__(self, policy: CompressionPolicy) -> None:
        self.policy = policy

    # ── Public API ────────────────────────────────────────────────────────────

    def should_compress(self, artifacts: list[CognitionArtifact]) -> bool:
        """Return True if artifact count exceeds policy max."""
        return len(artifacts) > self.policy.max_artifacts

    def compress(self, artifacts: list[CognitionArtifact]) -> ArtifactSummary:
        """
        Compress older artifacts into an ArtifactSummary.
        Preserves the most recent `preserve_recent` artifacts untouched.
        Applies policy filters before summarizing.
        """
        if not artifacts:
            raise ValueError("Cannot compress an empty artifact list.")

        sorted_artifacts = sorted(artifacts, key=lambda a: (a.round_id, a.role))
        compressible, _ = self._split(sorted_artifacts)

        if not compressible:
            raise ValueError(
                "No compressible artifacts after applying preserve_recent window."
            )

        filtered = self._apply_filters(compressible)
        content = self._build_content(filtered)

        return ArtifactSummary.create(
            source_roles=sorted({a.role for a in filtered}),
            rounds=sorted({a.round_id for a in filtered}),
            content=content or "(no compressible content after filtering)",
            metadata={
                "policy_max_artifacts": self.policy.max_artifacts,
                "policy_preserve_recent": self.policy.preserve_recent,
                "original_count": len(artifacts),
                "compressed_count": len(filtered),
            },
        )

    def split(
        self,
        artifacts: list[CognitionArtifact],
    ) -> tuple[list[CognitionArtifact], list[CognitionArtifact]]:
        """
        Public split: returns (compressible, preserved).
        Sorted by (round_id, role) before splitting.
        """
        sorted_artifacts = sorted(artifacts, key=lambda a: (a.round_id, a.role))
        return self._split(sorted_artifacts)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _split(
        self,
        sorted_artifacts: list[CognitionArtifact],
    ) -> tuple[list[CognitionArtifact], list[CognitionArtifact]]:
        """Split pre-sorted artifacts into (compressible, preserved) windows."""
        if self.policy.preserve_recent == 0:
            return sorted_artifacts, []
        return (
            sorted_artifacts[: -self.policy.preserve_recent],
            sorted_artifacts[-self.policy.preserve_recent :],
        )

    def _apply_filters(
        self,
        artifacts: list[CognitionArtifact],
    ) -> list[CognitionArtifact]:
        """Apply policy filters: empty content, deduplication, priority roles."""
        result = artifacts

        if self.policy.drop_empty_content:
            result = [a for a in result if self.policy.allows_content(a.content)]

        if self.policy.deduplicate:
            seen: set[str] = set()
            deduped: list[CognitionArtifact] = []
            for a in result:
                if a.artifact_id not in seen:
                    seen.add(a.artifact_id)
                    deduped.append(a)
            result = deduped

        if self.policy.has_priority_roles:
            priority = [a for a in result if self.policy.is_priority_role(a.role)]
            others = [a for a in result if not self.policy.is_priority_role(a.role)]
            result = priority + others

        return result

    def _build_content(self, artifacts: list[CognitionArtifact]) -> str:
        """Build compressed content string from filtered artifacts."""
        lines: list[str] = []
        total_chars = 0

        for artifact in artifacts:
            snippet = artifact.content[: self.policy.max_content_chars]
            line = f"[{artifact.role}] round={artifact.round_id}: {snippet}"

            if total_chars + len(line) > self.policy.max_content_chars:
                lines.append("[... compression limit reached ...]")
                break

            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)