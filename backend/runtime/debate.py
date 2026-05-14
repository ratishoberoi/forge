"""Deterministic courtroom-style orchestration across multiple candidates."""

from __future__ import annotations

from dataclasses import dataclass

from backend.runtime.candidate import PatchCandidate
from backend.runtime.consensus import ConsensusEngine, ConsensusResult
from backend.runtime.execution_review import ExecutionReview
from backend.runtime.judge import JudgeResult, PatchJudge
from backend.runtime.ranking import rank_candidates
from backend.runtime.reviewer import CandidateReview, CandidateReviewer


@dataclass(slots=True)
class DebateRound:
    round_index: int
    candidate_ids: tuple[str, ...]
    judge_results: dict[str, JudgeResult]
    reviews: dict[str, CandidateReview]
    ranking: tuple[str, ...]


@dataclass(slots=True)
class DebateResult:
    candidate_pool: tuple[PatchCandidate, ...]
    rounds: tuple[DebateRound, ...]
    consensus: ConsensusResult


class DebateOrchestrator:
    """Runs a single bounded multi-candidate debate round."""

    def __init__(
        self,
        *,
        judge: PatchJudge | None = None,
        reviewer: CandidateReviewer | None = None,
        consensus_engine: ConsensusEngine | None = None,
        max_candidates: int = 4,
    ) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be greater than zero.")
        self._judge = judge or PatchJudge()
        self._reviewer = reviewer or CandidateReviewer()
        self._consensus_engine = consensus_engine or ConsensusEngine()
        self._max_candidates = max_candidates

    def generate_candidate_pool(self, candidates: list[PatchCandidate]) -> list[PatchCandidate]:
        ordered = sorted(list(candidates), key=lambda candidate: (candidate.created_at, candidate.candidate_id))
        return ordered[: self._max_candidates]

    def review_candidates(
        self,
        candidates: list[PatchCandidate],
        *,
        judge_results: dict[str, JudgeResult] | None = None,
    ) -> dict[str, CandidateReview]:
        return {
            candidate.candidate_id: self._reviewer.review(
                candidate,
                (judge_results or {}).get(candidate.candidate_id),
            )
            for candidate in candidates
        }

    def compare_candidates(
        self,
        candidates: list[PatchCandidate],
        *,
        judge_results: dict[str, JudgeResult] | None = None,
        reviews: dict[str, CandidateReview] | None = None,
    ) -> list[PatchCandidate]:
        return rank_candidates(candidates, judge_results=judge_results, reviews=reviews)

    def select_consensus(
        self,
        candidates: list[PatchCandidate],
        *,
        judge_results: dict[str, JudgeResult] | None = None,
        reviews: dict[str, CandidateReview] | None = None,
    ) -> ConsensusResult:
        return self._consensus_engine.select(
            candidates,
            judge_results=judge_results,
            reviews=reviews,
        )

    def run_debate(
        self,
        candidates: list[PatchCandidate],
        *,
        execution_reviews: dict[str, ExecutionReview] | None = None,
    ) -> DebateResult:
        candidate_pool = self.generate_candidate_pool(candidates)
        judge_results = {
            result.candidate_id: result
            for result in self._judge.evaluate_candidates(
                candidate_pool,
                execution_reviews=execution_reviews,
            )
        }
        reviews = self.review_candidates(candidate_pool, judge_results=judge_results)
        ranking = self.compare_candidates(candidate_pool, judge_results=judge_results, reviews=reviews)
        consensus = self.select_consensus(candidate_pool, judge_results=judge_results, reviews=reviews)
        round_result = DebateRound(
            round_index=1,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidate_pool),
            judge_results=judge_results,
            reviews=reviews,
            ranking=tuple(candidate.candidate_id for candidate in ranking),
        )
        return DebateResult(
            candidate_pool=tuple(candidate_pool),
            rounds=(round_result,),
            consensus=consensus,
        )
