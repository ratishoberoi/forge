from __future__ import annotations
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_timeline import ArtifactTimeline
from backend.runtime.timeline_policy import TimelinePolicy


class ArtifactReplayEngine:
    """
    Replays ArtifactTimeline through a TimelinePolicy lens.
    Responsibilities:
    - apply max_replay_artifacts cap
    - enforce preserve_recent window
    - filter empty content and duplicates
    - surface priority roles first
    - apply ordering strategy
    - enforce content character budget
    """

    def __init__(self, policy: TimelinePolicy) -> None:
        self.policy = policy

    # ── Public API ────────────────────────────────────────────────────────────

    def replay(self, timeline: ArtifactTimeline) -> list[CognitionArtifact]:
        """
        Apply full policy to timeline and return replay-ready artifact list.
        Stages:
        1. Order by policy sort key.
        2. Apply content filters (empty, dedup).
        3. Split into compressible + preserved windows.
        4. Apply priority role ordering.
        5. Cap to max_replay_artifacts.
        6. Enforce content character budget.
        """
        if timeline.is_empty:
            return []

        ordered = self._order(timeline)
        filtered = self._filter(ordered)
        compressible, preserved = self._split(filtered)
        compressible = self._apply_priority(compressible)
        candidates = compressible + preserved
        capped = candidates[-self.policy.max_replay_artifacts:]
        return self._enforce_char_budget(capped)

    def replay_for_role(
        self,
        timeline: ArtifactTimeline,
        role: str,
    ) -> list[CognitionArtifact]:
        """Replay filtered to a specific role."""
        filtered = [a for a in self.replay(timeline) if a.role.lower() == role.lower()]
        return filtered

    def replay_round_range(
        self,
        timeline: ArtifactTimeline,
        *,
        start: int,
        end: int,
    ) -> list[CognitionArtifact]:
        """Replay filtered to a specific round range [start, end] inclusive."""
        return [a for a in self.replay(timeline) if start <= a.round_id <= end]

    def replay_latest(
        self,
        timeline: ArtifactTimeline,
        *,
        n: int,
    ) -> list[CognitionArtifact]:
        """Replay the n most recent artifacts after policy is applied."""
        return self.replay(timeline)[-n:]

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def _order(self, timeline: ArtifactTimeline) -> list[CognitionArtifact]:
        """Sort artifacts by policy order_by strategy."""
        return sorted(timeline.artifacts, key=self.policy.sort_key())

    def _filter(
        self, artifacts: list[CognitionArtifact]
    ) -> list[CognitionArtifact]:
        """Apply drop_empty_content and deduplicate filters."""
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

        return result

    def _split(
        self,
        artifacts: list[CognitionArtifact],
    ) -> tuple[list[CognitionArtifact], list[CognitionArtifact]]:
        """Split into (compressible, preserved) windows."""
        if self.policy.preserve_recent == 0:
            return artifacts, []
        return (
            artifacts[: -self.policy.preserve_recent],
            artifacts[-self.policy.preserve_recent:],
        )

    def _apply_priority(
        self,
        artifacts: list[CognitionArtifact],
    ) -> list[CognitionArtifact]:
        """Surface priority role artifacts before others."""
        if not self.policy.has_priority_roles:
            return artifacts
        priority = [a for a in artifacts if self.policy.is_priority_role(a.role)]
        others = [a for a in artifacts if not self.policy.is_priority_role(a.role)]
        return priority + others

    def _enforce_char_budget(
        self,
        artifacts: list[CognitionArtifact],
    ) -> list[CognitionArtifact]:
        """Drop artifacts that exceed cumulative max_content_chars budget."""
        result: list[CognitionArtifact] = []
        total = 0
        for artifact in artifacts:
            if total + len(artifact.content) > self.policy.max_content_chars:
                break
            result.append(artifact)
            total += len(artifact.content)
        return result