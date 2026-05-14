"""Patch candidate collection and ranking."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.runtime.patches import StructuredPatch

if TYPE_CHECKING:
    from backend.runtime.judge import JudgeResult


@dataclass(slots=True)
class PatchCandidate:
    patch: StructuredPatch
    agent_id: str
    candidate_id: str = field(default_factory=lambda: f"candidate-{uuid.uuid4().hex}")
    created_at: float = field(default_factory=time.time)
    judge_result: JudgeResult | None = None
    critique: str | None = None

    @property
    def score(self) -> float | None:
        if self.judge_result is None:
            return None
        return self.judge_result.score.composite_score


class CandidateCollection:
    def __init__(self) -> None:
        self._candidates: list[PatchCandidate] = []

    def add_candidate(self, candidate: PatchCandidate) -> None:
        self._candidates.append(candidate)

    def attach_judge_result(self, candidate_id: str, judge_result: JudgeResult) -> None:
        candidate = self._find(candidate_id)
        candidate.judge_result = judge_result
        candidate.critique = judge_result.critique_summary

    def rank_candidates(self) -> list[PatchCandidate]:
        return sorted(
            self._candidates,
            key=lambda candidate: (
                -(candidate.score if candidate.score is not None else float("-inf")),
                candidate.created_at,
                candidate.candidate_id,
            ),
        )

    def get_best_candidate(self) -> PatchCandidate | None:
        ranked = self.rank_candidates()
        return ranked[0] if ranked else None

    def all(self) -> list[PatchCandidate]:
        return list(self._candidates)

    def _find(self, candidate_id: str) -> PatchCandidate:
        for candidate in self._candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(f"Unknown candidate_id '{candidate_id}'.")
