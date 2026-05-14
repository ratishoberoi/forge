from __future__ import annotations

from backend.runtime.candidate import PatchCandidate
from backend.runtime.consensus import ConsensusEngine
from backend.runtime.judge import JudgeResult, JudgeScore
from backend.runtime.patches import StructuredPatch
from backend.runtime.reviewer import CandidateReview


def make_candidate(candidate_id: str) -> PatchCandidate:
    return PatchCandidate(
        candidate_id=candidate_id,
        agent_id="agent-1",
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
        critique_summary=f"review-{candidate_id}",
        weaknesses=(),
        architecture_risk="low",
        maintainability_score=maintainability,
        hallucination_risk=hallucination,
        execution_risk=execution,
        adversarial_question="question",
    )


def test_consensus_selects_stable_winner_and_aggregates_critiques() -> None:
    engine = ConsensusEngine()
    first = make_candidate("candidate-a")
    second = make_candidate("candidate-b")

    result = engine.select(
        [second, first],
        judge_results={
            "candidate-a": make_judge_result("candidate-a", 8.5),
            "candidate-b": make_judge_result("candidate-b", 7.5),
        },
        reviews={
            "candidate-a": make_review("candidate-a", maintainability=8.0, hallucination=2.0, execution=1.0),
            "candidate-b": make_review("candidate-b", maintainability=7.0, hallucination=4.0, execution=3.0),
        },
    )

    assert result.winner is first
    assert [candidate.candidate_id for candidate in result.ranked_candidates] == ["candidate-a", "candidate-b"]
    assert result.confidence > 0.5
    assert result.critique_summaries["candidate-a"] == "review-candidate-a"


def test_consensus_marks_tie_breaking_deterministically() -> None:
    engine = ConsensusEngine()
    first = make_candidate("candidate-a")
    second = make_candidate("candidate-b")

    result = engine.select(
        [second, first],
        judge_results={
            "candidate-a": make_judge_result("candidate-a", 8.0),
            "candidate-b": make_judge_result("candidate-b", 8.0),
        },
    )

    assert result.winner is first
    assert result.tie_broken is True

