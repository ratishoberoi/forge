from __future__ import annotations

from backend.runtime.candidate import PatchCandidate
from backend.runtime.debate import DebateOrchestrator
from backend.runtime.execution import ExecutionResult
from backend.runtime.execution_review import build_execution_summary
from backend.runtime.patches import PatchRisk, PatchTarget, StructuredPatch


def make_candidate(
    candidate_id: str,
    *,
    created_at: float,
    risk: PatchRisk = PatchRisk.LOW,
    impacted_files: int = 1,
) -> PatchCandidate:
    return PatchCandidate(
        candidate_id=candidate_id,
        agent_id="agent-1",
        created_at=created_at,
        patch=StructuredPatch(
            title=candidate_id,
            unified_diff="--- a/app.py\n+++ b/app.py\n@@\n+print('x')\n",
            impacted_files=[PatchTarget(path=f"file-{index}.py") for index in range(impacted_files)],
            risk=risk,
            summary="summary",
            reasoning="reasoning",
        ),
    )


def test_debate_bounds_candidate_pool_and_returns_single_round() -> None:
    orchestrator = DebateOrchestrator(max_candidates=2)
    candidates = [
        make_candidate("candidate-c", created_at=3.0),
        make_candidate("candidate-a", created_at=1.0),
        make_candidate("candidate-b", created_at=2.0),
    ]

    result = orchestrator.run_debate(candidates)

    assert [candidate.candidate_id for candidate in result.candidate_pool] == ["candidate-a", "candidate-b"]
    assert len(result.rounds) == 1
    assert result.rounds[0].candidate_ids == ("candidate-a", "candidate-b")


def test_debate_preserves_candidate_immutability_and_selects_consensus() -> None:
    orchestrator = DebateOrchestrator(max_candidates=3)
    candidate_a = make_candidate("candidate-a", created_at=1.0)
    candidate_b = make_candidate("candidate-b", created_at=2.0, risk=PatchRisk.HIGH, impacted_files=3)
    original_a_critique = candidate_a.critique
    original_b_critique = candidate_b.critique

    result = orchestrator.run_debate(
        [candidate_b, candidate_a],
        execution_reviews={
            "candidate-a": build_execution_summary(
                ExecutionResult(
                    tool="pytest",
                    success=True,
                    exit_code=0,
                    stdout="passed",
                    stderr="",
                    duration=0.2,
                    timed_out=False,
                )
            )
        },
    )

    assert result.consensus.winner is candidate_a
    assert result.rounds[0].ranking[0] == "candidate-a"
    assert candidate_a.critique == original_a_critique
    assert candidate_b.critique == original_b_critique
