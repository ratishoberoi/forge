from __future__ import annotations

from backend.runtime.candidate import PatchCandidate
from backend.runtime.execution import ExecutionResult
from backend.runtime.execution_review import build_execution_summary
from backend.runtime.judge import PatchJudge
from backend.runtime.patches import PatchRisk, PatchTarget, StructuredPatch
from backend.runtime.reviewer import CandidateReviewer


def make_candidate(
    candidate_id: str,
    *,
    risk: PatchRisk = PatchRisk.LOW,
    impacted_files: int = 1,
    validation_errors: list[str] | None = None,
    reasoning: str | None = "reasoning",
    summary: str | None = "summary",
) -> PatchCandidate:
    return PatchCandidate(
        candidate_id=candidate_id,
        agent_id="agent-1",
        patch=StructuredPatch(
            title=candidate_id,
            unified_diff="--- a/app.py\n+++ b/app.py\n@@\n+print('x')\n",
            impacted_files=[PatchTarget(path=f"file-{index}.py") for index in range(impacted_files)],
            risk=risk,
            validation_errors=validation_errors or [],
            reasoning=reasoning,
            summary=summary,
        ),
    )


def test_reviewer_generates_adversarial_critique_and_risks() -> None:
    reviewer = CandidateReviewer()
    candidate = make_candidate(
        "candidate-1",
        risk=PatchRisk.HIGH,
        impacted_files=3,
        validation_errors=["syntax error"],
        reasoning=None,
        summary=None,
    )

    review = reviewer.review(candidate)

    assert review.candidate_id == "candidate-1"
    assert review.weaknesses
    assert review.architecture_risk in {"medium", "high"}
    assert review.hallucination_risk > 2.0
    assert "candidate-1" in review.adversarial_question


def test_reviewer_uses_execution_evidence_without_mutating_candidate() -> None:
    reviewer = CandidateReviewer()
    judge = PatchJudge()
    candidate = make_candidate("candidate-2")
    original_summary = candidate.patch.summary
    execution_review = build_execution_summary(
        ExecutionResult(
            tool="pytest",
            success=False,
            exit_code=1,
            stdout="",
            stderr="Traceback: failure",
            duration=0.4,
            timed_out=False,
        )
    )
    judge_result = judge.evaluate_candidate(candidate, execution_review=execution_review)

    review = reviewer.review(candidate, judge_result)

    assert review.execution_risk >= 5.5
    assert "Traceback" in review.critique_summary
    assert candidate.patch.summary == original_summary

