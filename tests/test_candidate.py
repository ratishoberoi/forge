from __future__ import annotations

from backend.runtime.candidate import CandidateCollection, PatchCandidate
from backend.runtime.judge import JudgeResult, JudgeScore
from backend.runtime.patches import StructuredPatch


def make_candidate(title: str, score_value: float, candidate_id: str) -> PatchCandidate:
    candidate = PatchCandidate(
        candidate_id=candidate_id,
        agent_id="agent-1",
        patch=StructuredPatch(title=title, unified_diff="--- a/x\n+++ b/x\n@@\n+change"),
    )
    candidate.judge_result = JudgeResult(
        candidate_id=candidate.candidate_id,
        score=JudgeScore(
            correctness=score_value,
            architecture=score_value,
            safety=score_value,
            minimality=score_value,
            maintainability=score_value,
            hallucination_risk=0.0,
        ),
        reasoning="ok",
        critique_summary="critique",
        recommendation="accept",
        retry_recommended=False,
        winning_candidate_id=candidate.candidate_id,
    )
    return candidate


def test_candidate_collection_ranks_highest_score_first() -> None:
    collection = CandidateCollection()
    low = make_candidate("low", 4.0, "candidate-b")
    high = make_candidate("high", 9.0, "candidate-a")
    collection.add_candidate(low)
    collection.add_candidate(high)

    ranked = collection.rank_candidates()
    assert ranked[0].candidate_id == "candidate-a"
    assert collection.get_best_candidate() is high


def test_candidate_collection_attach_judge_result_and_critique() -> None:
    collection = CandidateCollection()
    candidate = PatchCandidate(
        candidate_id="candidate-1",
        agent_id="agent-1",
        patch=StructuredPatch(title="x", unified_diff="--- a/x\n+++ b/x\n@@\n+change"),
    )
    collection.add_candidate(candidate)
    result = JudgeResult(
        candidate_id="candidate-1",
        score=JudgeScore(8, 8, 8, 8, 8, 2),
        reasoning="solid",
        critique_summary="safe patch",
        recommendation="accept",
        retry_recommended=False,
        winning_candidate_id="candidate-1",
    )

    collection.attach_judge_result("candidate-1", result)

    assert candidate.judge_result is result
    assert candidate.critique == "safe patch"
