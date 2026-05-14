from __future__ import annotations

from backend.runtime.candidate import PatchCandidate
from backend.runtime.judge import JudgeResult, JudgeScore
from backend.runtime.patches import PatchRisk, StructuredPatch
from backend.runtime.retry import RetryOrchestrator, RetryPolicy


def make_candidate(candidate_id: str, score: float, *, retry_recommended: bool = True) -> tuple[PatchCandidate, JudgeResult]:
    candidate = PatchCandidate(
        candidate_id=candidate_id,
        agent_id="agent-1",
        patch=StructuredPatch(
            title=f"patch-{candidate_id}",
            summary="summary",
            reasoning="reasoning",
            unified_diff="--- a/app.py\n+++ b/app.py\n@@\n+print('x')\n",
            risk=PatchRisk.LOW,
        ),
    )
    result = JudgeResult(
        candidate_id=candidate_id,
        score=JudgeScore(
            correctness=score,
            architecture=score,
            safety=score,
            minimality=score,
            maintainability=score,
            hallucination_risk=max(0.0, 10.0 - score),
        ),
        reasoning="judge reasoning",
        critique_summary="judge critique",
        recommendation="accept" if not retry_recommended else "revise",
        retry_recommended=retry_recommended,
        winning_candidate_id=candidate_id,
    )
    return candidate, result


def test_retry_orchestrator_stops_when_judge_accepts() -> None:
    orchestrator = RetryOrchestrator()
    candidate, result = make_candidate("candidate-1", 9.0, retry_recommended=False)

    decision = orchestrator.decide_retry(
        task="fix auth caching",
        candidate=candidate,
        judge_result=result,
        repository_context="auth flow",
    )

    assert decision.should_retry is False
    assert decision.stop_reason == "accepted"


def test_retry_orchestrator_respects_retry_threshold() -> None:
    orchestrator = RetryOrchestrator(policy=RetryPolicy(max_retries=3, min_score_improvement=0.5))
    first_candidate, first_result = make_candidate("candidate-1", 6.0, retry_recommended=True)
    second_candidate, second_result = make_candidate("candidate-2", 6.2, retry_recommended=True)
    third_candidate, third_result = make_candidate("candidate-3", 6.25, retry_recommended=True)

    first_decision = orchestrator.decide_retry(
        task="fix auth caching",
        candidate=first_candidate,
        judge_result=first_result,
        repository_context="auth flow",
    )
    second_decision = orchestrator.decide_retry(
        task="fix auth caching",
        candidate=second_candidate,
        judge_result=second_result,
        repository_context="auth flow",
    )
    third_decision = orchestrator.decide_retry(
        task="fix auth caching",
        candidate=third_candidate,
        judge_result=third_result,
        repository_context="auth flow",
    )

    assert first_decision.should_retry is True
    assert second_decision.should_retry is True
    assert third_decision.should_retry is False
    assert third_decision.stop_reason == "stagnation_detected"


def test_retry_orchestrator_tracks_history_and_best_candidate() -> None:
    orchestrator = RetryOrchestrator(policy=RetryPolicy(max_retries=2))
    first_candidate, first_result = make_candidate("candidate-1", 6.0, retry_recommended=True)
    second_candidate, second_result = make_candidate("candidate-2", 7.0, retry_recommended=True)

    orchestrator.decide_retry(
        task="fix auth caching",
        candidate=first_candidate,
        judge_result=first_result,
        repository_context="auth flow",
    )
    orchestrator.decide_retry(
        task="fix auth caching",
        candidate=second_candidate,
        judge_result=second_result,
        repository_context="auth flow",
    )
    final = orchestrator.finalize()

    assert final.history == ["candidate-1", "candidate-2"]
    assert final.best_candidate.candidate_id == "candidate-2"
    assert final.retry_count == 1


def test_retry_orchestrator_stops_at_retry_limit_deterministically() -> None:
    orchestrator = RetryOrchestrator(policy=RetryPolicy(max_retries=1, min_score_improvement=0.1))
    first_candidate, first_result = make_candidate("candidate-1", 6.0, retry_recommended=True)
    second_candidate, second_result = make_candidate("candidate-2", 6.5, retry_recommended=True)

    first = orchestrator.decide_retry(
        task="fix auth caching",
        candidate=first_candidate,
        judge_result=first_result,
        repository_context="auth flow",
    )
    second = orchestrator.decide_retry(
        task="fix auth caching",
        candidate=second_candidate,
        judge_result=second_result,
        repository_context="auth flow",
    )

    assert first.should_retry is True
    assert second.should_retry is False
    assert second.stop_reason == "retry_limit_reached"
