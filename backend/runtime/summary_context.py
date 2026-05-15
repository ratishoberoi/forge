from __future__ import annotations
from backend.runtime.artifact_summary import ArtifactSummary


class SummaryContextBuilder:
    """
    Builds LLM-readable context strings from ArtifactSummaries.
    Responsibilities:
    - format summaries into prompt-injectable context blocks
    - filter by role, round, or recency
    - enforce character budgets
    - produce debug-friendly representations
    """

    DEFAULT_SEPARATOR = "\n\n"
    DEFAULT_MAX_CHARS = 32_000

    # ── Core build ────────────────────────────────────────────────────────────

    def build(
        self,
        summaries: list[ArtifactSummary],
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """Format all summaries into a single context block."""
        if not summaries:
            return ""

        sections = [self._format(s) for s in summaries]
        context = self.DEFAULT_SEPARATOR.join(sections)

        if len(context) > max_chars:
            context = context[:max_chars]
            context += "\n\n[... summary context truncated ...]"

        return context

    def build_for_role(
        self,
        summaries: list[ArtifactSummary],
        role: str,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """Build context from summaries that include a specific role."""
        filtered = [s for s in summaries if s.covers_role(role)]
        return self.build(filtered, max_chars=max_chars)

    def build_for_round(
        self,
        summaries: list[ArtifactSummary],
        round_id: int,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """Build context from summaries that cover a specific round_id."""
        filtered = [s for s in summaries if s.covers_round(round_id)]
        return self.build(filtered, max_chars=max_chars)

    def build_latest(
        self,
        summaries: list[ArtifactSummary],
        *,
        n: int = 3,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """Build context from the n most recently created summaries."""
        sorted_summaries = sorted(
            summaries,
            key=lambda s: s.created_at,
            reverse=True,
        )[:n]
        sorted_summaries = sorted(sorted_summaries, key=lambda s: s.created_at)
        return self.build(sorted_summaries, max_chars=max_chars)

    def build_combined(
        self,
        summaries: list[ArtifactSummary],
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """
        Merge all summary contents into a single condensed block.
        No per-summary headers — used when context space is tight.
        """
        if not summaries:
            return ""

        combined = "\n".join(s.content for s in summaries)
        if len(combined) > max_chars:
            combined = combined[:max_chars]
            combined += "\n[... combined context truncated ...]"
        return combined

    def summarize(self, summaries: list[ArtifactSummary]) -> str:
        """
        One-line debug representation per summary.
        Does not truncate — for logging only.
        """
        if not summaries:
            return "(no summaries)"
        lines = [
            f"id={s.summary_id[:8]} "
            f"roles={s.source_roles} "
            f"rounds={s.rounds} "
            f"chars={len(s.content)} "
            f"span={s.round_span}"
            for s in summaries
        ]
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format(summary: ArtifactSummary) -> str:
        """Format a single summary into a context block."""
        min_round, max_round = summary.round_span
        return (
            f"[SUMMARY {summary.summary_id[:8]}] "
            f"[ROLES: {', '.join(summary.source_roles)}] "
            f"[ROUNDS: {min_round}-{max_round}]\n"
            f"{summary.content}"
        )