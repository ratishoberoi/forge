from __future__ import annotations
from backend.runtime.artifact import CognitionArtifact


class ArtifactContextBuilder:
    """
    Builds prompt context strings from CognitionArtifacts.
    Responsibilities:
    - format artifact history into LLM-readable context
    - filter by role or round range
    - truncate context to token budget
    - summarize artifact sets
    """

    DEFAULT_SEPARATOR = "\n\n"
    DEFAULT_MAX_CHARS = 32_000  # ~8k tokens at 4 chars/token

    def build_context(
        self,
        artifacts: list[CognitionArtifact],
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """
        Format artifacts into a single LLM-readable context block.
        Truncates total output to max_chars to respect token budgets.
        """
        if not artifacts:
            return ""

        sections = [self._format(a) for a in artifacts]
        context = self.DEFAULT_SEPARATOR.join(sections)

        if len(context) > max_chars:
            context = context[:max_chars]
            context += "\n\n[... context truncated ...]"

        return context

    def build_role_context(
        self,
        artifacts: list[CognitionArtifact],
        role: str,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """Build context filtered to a single role."""
        filtered = [a for a in artifacts if a.role.lower() == role.lower()]
        return self.build_context(filtered, max_chars=max_chars)

    def build_round_range_context(
        self,
        artifacts: list[CognitionArtifact],
        *,
        start: int,
        end: int,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """Build context filtered to a round_id range [start, end] inclusive."""
        filtered = [a for a in artifacts if start <= a.round_id <= end]
        return self.build_context(filtered, max_chars=max_chars)

    def build_latest_context(
        self,
        artifacts: list[CognitionArtifact],
        *,
        n: int = 3,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """Build context from the n most recent artifacts by round_id."""
        latest = sorted(artifacts, key=lambda a: a.round_id, reverse=True)[:n]
        latest = sorted(latest, key=lambda a: a.round_id)
        return self.build_context(latest, max_chars=max_chars)

    def summarize(self, artifacts: list[CognitionArtifact]) -> str:
        """
        One-line summary per artifact — for logging or debug output.
        Does not truncate.
        """
        if not artifacts:
            return "(no artifacts)"
        lines = [
            f"round={a.round_id} role={a.role} chars={len(a.content)} task={a.task!r}"
            for a in sorted(artifacts, key=lambda a: a.round_id)
        ]
        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _format(artifact: CognitionArtifact) -> str:
        """Format a single artifact into a context block."""
        return (
            f"[ROLE: {artifact.role}] [ROUND: {artifact.round_id}] "
            f"[TASK: {artifact.task}]\n"
            f"{artifact.content}"
        )