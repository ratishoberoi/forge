"""Consensus selection across multiple patch candidates."""

from __future__ import annotations

from dataclasses import dataclass

from backend.runtime.candidate import PatchCandidate
from backend.runtime.judge import JudgeResult
from backend.runtime.ranking import build_score_table, rank_candidates
from backend.runtime.reviewer import CandidateReview


@dataclass(slots=True)
class ConsensusResult:
    winner: PatchCandidate | None
    ranked_candidates: tuple[PatchCandidate, ...]
    confidence: float
    score_table: dict[str, float]
    critique_summaries: dict[str, str]
    tie_broken: bool


class ConsensusEngine:
    """Aggregates judge and reviewer signals into a stable winner selection."""

    def select(
        self,
        candidates: list[PatchCandidate],
        *,
        judge_results: dict[str, JudgeResult] | None = None,
        reviews: dict[str, CandidateReview] | None = None,
    ) -> ConsensusResult:
        ranked_candidates = tuple(
            rank_candidates(candidates, judge_results=judge_results, reviews=reviews)
        )
        score_table = build_score_table(candidates, judge_results=judge_results, reviews=reviews)
        critique_summaries = {
            candidate_id: review.critique_summary for candidate_id, review in (reviews or {}).items()
        }

        winner = ranked_candidates[0] if ranked_candidates else None
        tie_broken = self._tie_broken(ranked_candidates, score_table)
        confidence = self._confidence(ranked_candidates, score_table)

        return ConsensusResult(
            winner=winner,
            ranked_candidates=ranked_candidates,
            confidence=confidence,
            score_table=score_table,
            critique_summaries=critique_summaries,
            tie_broken=tie_broken,
        )

    @staticmethod
    def _confidence(
        ranked_candidates: tuple[PatchCandidate, ...],
        score_table: dict[str, float],
    ) -> float:
        if not ranked_candidates:
            return 0.0
        if len(ranked_candidates) == 1:
            return 1.0
        winner_score = score_table[ranked_candidates[0].candidate_id]
        runner_up_score = score_table[ranked_candidates[1].candidate_id]
        delta = max(0.0, winner_score - runner_up_score)
        return round(min(1.0, 0.5 + (delta / 4.0)), 3)

    @staticmethod
    def _tie_broken(
        ranked_candidates: tuple[PatchCandidate, ...],
        score_table: dict[str, float],
    ) -> bool:
        if len(ranked_candidates) < 2:
            return False
        winner_score = score_table[ranked_candidates[0].candidate_id]
        runner_up_score = score_table[ranked_candidates[1].candidate_id]
        return winner_score == runner_up_score
