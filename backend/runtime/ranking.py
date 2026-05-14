"""Deterministic ranking utilities for patch candidates."""

from __future__ import annotations

from backend.runtime.candidate import PatchCandidate
from backend.runtime.judge import JudgeResult
from backend.runtime.reviewer import CandidateReview


def compare_scores(left: float, right: float) -> int:
    """Compare two numeric scores with deterministic three-way semantics."""
    if left > right:
        return 1
    if left < right:
        return -1
    return 0


def rank_candidates(
    candidates: list[PatchCandidate],
    *,
    judge_results: dict[str, JudgeResult] | None = None,
    reviews: dict[str, CandidateReview] | None = None,
) -> list[PatchCandidate]:
    """Return a new deterministically ordered candidate list."""
    score_table = build_score_table(candidates, judge_results=judge_results, reviews=reviews)
    review_table = reviews or {}
    return sorted(
        list(candidates),
        key=lambda candidate: (
            -score_table[candidate.candidate_id],
            review_table.get(candidate.candidate_id).hallucination_risk
            if candidate.candidate_id in review_table
            else float("inf"),
            candidate.created_at,
            candidate.candidate_id,
        ),
    )


def build_score_table(
    candidates: list[PatchCandidate],
    *,
    judge_results: dict[str, JudgeResult] | None = None,
    reviews: dict[str, CandidateReview] | None = None,
) -> dict[str, float]:
    """Build composite candidate scores without mutating candidates."""
    judge_results = judge_results or {}
    reviews = reviews or {}
    score_table: dict[str, float] = {}
    for candidate in candidates:
        judge_score = _judge_score(candidate, judge_results)
        review_score = reviews.get(candidate.candidate_id).review_score if candidate.candidate_id in reviews else None
        if review_score is None:
            combined = judge_score
        elif judge_score is None:
            combined = review_score
        else:
            combined = (judge_score * 0.7) + (review_score * 0.3)
        score_table[candidate.candidate_id] = round(combined if combined is not None else 0.0, 3)
    return score_table


def _judge_score(candidate: PatchCandidate, judge_results: dict[str, JudgeResult]) -> float | None:
    if candidate.candidate_id in judge_results:
        return judge_results[candidate.candidate_id].score.composite_score
    if candidate.judge_result is not None:
        return candidate.judge_result.score.composite_score
    return None
