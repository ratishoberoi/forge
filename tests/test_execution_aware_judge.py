from __future__ import annotations

from backend.runtime.candidate import PatchCandidate
from backend.runtime.execution import ExecutionResult
from backend.runtime.execution_review import ExecutionSeverity, build_execution_summary
from backend.runtime.judge import PatchJudge
from backend.runtime.patches import StructuredPatch
from backend.runtime.retry import RetryOrchestrator


def make_candidate(candidate_id: str) -> PatchCandidate:
    return PatchCandidate(
        candidate_id=candidate_id,
        agent_id="agent-1",
        patch=StructuredPatch(
            title="candidate patch",
            summary="summary",
            reasoning="reasoning",
            unified_diff="--- a/app.py\n+++ b/app.py\n@@\n+print('x')\n",
        ),
    )


def test_judge_increases_score_on_successful_execution() -> None:
    judge = PatchJudge()
    candidate = make_candidate("candidate-1")
    baseline = judge.evaluate_candidate(candidate)
    execution_review = build_execution_summary(
        ExecutionResult(
            tool="pytest",
            success=True,
            exit_code=0,
            stdout="1 passed",
            stderr="",
            duration=0.2,
            timed_out=False,
        )
    )

    result = judge.evaluate_candidate(candidate, execution_review)

    assert result.score.correctness > baseline.score.correctness
    assert result.execution_review is not None


def test_judge_reduces_score_on_failed_execution() -> None:
    judge = PatchJudge()
    candidate = make_candidate("candidate-1")
    baseline = judge.evaluate_candidate(candidate)
    execution_review = build_execution_summary(
        ExecutionResult(
            tool="pytest",
            success=False,
            exit_code=1,
            stdout="F",
            stderr="Traceback\nAssertionError: broken",
            duration=0.2,
            timed_out=False,
        )
    )

    result = judge.evaluate_candidate(candidate, execution_review)

    assert result.score.correctness < baseline.score.correctness
    assert result.score.maintainability < baseline.score.maintainability


def test_judge_timeout_penalty_increases_hallucination_risk() -> None:
    judge = PatchJudge()
    candidate = make_candidate("candidate-1")
    baseline = judge.evaluate_candidate(candidate)
    execution_review = build_execution_summary(
        ExecutionResult(
            tool="pytest",
            success=False,
            exit_code=None,
            stdout="",
            stderr="Execution timed out after 1 seconds.",
            duration=1.0,
            timed_out=True,
        )
    )

    result = judge.evaluate_candidate(candidate, execution_review)

    assert execution_review.severity is ExecutionSeverity.CRITICAL
    assert result.score.hallucination_risk > baseline.score.hallucination_risk
    assert result.retry_priority == "high"


def test_retry_orchestrator_escalates_critical_execution_failures() -> None:
    judge = PatchJudge()
    orchestrator = RetryOrchestrator()
    candidate = make_candidate("candidate-1")
    execution_review = build_execution_summary(
        ExecutionResult(
            tool="pytest",
            success=False,
            exit_code=None,
            stdout="",
            stderr="Execution timed out after 1 seconds.",
            duration=1.0,
            timed_out=True,
        )
    )
    judged = judge.evaluate_candidate(candidate, execution_review)

    decision = orchestrator.decide_retry(
        task="fix auth caching",
        candidate=candidate,
        judge_result=judged,
        repository_context="auth flow",
    )

    assert decision.should_retry is True
    assert decision.reason == "critical_execution_failure"
    assert decision.retry_priority == "high"
    assert decision.execution_aware is True


def test_retry_orchestrator_stops_on_repeated_identical_execution_failures() -> None:
    judge = PatchJudge()
    orchestrator = RetryOrchestrator()
    review = build_execution_summary(
        ExecutionResult(
            tool="pytest",
            success=False,
            exit_code=1,
            stdout="",
            stderr="Traceback\nAssertionError: same failure",
            duration=0.2,
            timed_out=False,
        )
    )

    first = judge.evaluate_candidate(make_candidate("candidate-1"), review)
    second = judge.evaluate_candidate(make_candidate("candidate-2"), review)

    first_decision = orchestrator.decide_retry(
        task="fix auth caching",
        candidate=make_candidate("candidate-1"),
        judge_result=first,
        repository_context="auth flow",
    )
    second_decision = orchestrator.decide_retry(
        task="fix auth caching",
        candidate=make_candidate("candidate-2"),
        judge_result=second,
        repository_context="auth flow",
    )

    assert first_decision.should_retry is True
    assert second_decision.should_retry is False
    assert second_decision.stop_reason == "repeated_execution_failure"
