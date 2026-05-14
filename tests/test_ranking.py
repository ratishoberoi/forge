from __future__ import annotations

from backend.runtime.candidate import PatchCandidate
from backend.runtime.judge import JudgeResult, JudgeScore
from backend.runtime.patches import StructuredPatch
from backend.runtime.ranking import compare_scores, rank_candidates
from backend.runtime.reviewer import CandidateReview


def make_candidate(candidate_id: str, *, created_at: float) -> PatchCandidate:
    return PatchCandidate(
        candidate_id=candidate_id,
        agent_id="agent-1",
        created_at=created_at,
        patch=StructuredPatch(title=candidate_id, unified_diff="--- a/x\n+++ b/x\n@@\n+change"),
    )


def make_judge_result(candidate_id: str, score_value: float) -> JudgeResult:
    return JudgeResult(
        candidate_id=candidate_id,
        score=JudgeScore(score_value, score_value, score_value, score_value, score_value, 2.0),
        reasoning="reasoning",
        critique_summary="critique",
        recommendation="accept",
        retry_recommended=False,
    )


def make_review(candidate_id: str, *, maintainability: float, hallucination: float, execution: float) -> CandidateReview:
    return CandidateReview(
        candidate_id=candidate_id,
        critique_summary="review",
        weaknesses=(),
        architecture_risk="low",
        maintainability_score=maintainability,
        hallucination_risk=hallucination,
        execution_risk=execution,
        adversarial_question="question",
    )


def test_compare_scores_has_deterministic_three_way_semantics() -> None:
    assert compare_scores(2.0, 1.0) == 1
    assert compare_scores(1.0, 2.0) == -1
    assert compare_scores(1.0, 1.0) == 0


def test_rank_candidates_prefers_higher_combined_score() -> None:
    first = make_candidate("candidate-a", created_at=2.0)
    second = make_candidate("candidate-b", created_at=1.0)

    ranked = rank_candidates(
        [first, second],
        judge_results={
            "candidate-a": make_judge_result("candidate-a", 8.5),
            "candidate-b": make_judge_result("candidate-b", 7.0),
        },
        reviews={
            "candidate-a": make_review("candidate-a", maintainability=8.0, hallucination=2.0, execution=1.0),
            "candidate-b": make_review("candidate-b", maintainability=7.0, hallucination=4.0, execution=5.0),
        },
    )

    assert [candidate.candidate_id for candidate in ranked] == ["candidate-a", "candidate-b"]


def test_rank_candidates_breaks_ties_stably() -> None:
    first = make_candidate("candidate-a", created_at=1.0)
    second = make_candidate("candidate-b", created_at=1.0)

    ranked = rank_candidates(
        [second, first],
        judge_results={
            "candidate-a": make_judge_result("candidate-a", 8.0),
            "candidate-b": make_judge_result("candidate-b", 8.0),
        },
    )

    assert [candidate.candidate_id for candidate in ranked] == ["candidate-a", "candidate-b"]

