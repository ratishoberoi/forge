from __future__ import annotations

from backend.runtime.candidate import CandidateCollection, PatchCandidate
from backend.runtime.judge import PatchJudge
from backend.runtime.patches import PatchRisk, PatchTarget, StructuredPatch


def make_patch_candidate(
    candidate_id: str,
    *,
    risk: PatchRisk = PatchRisk.LOW,
    impacted_files: int = 1,
    validation_errors: list[str] | None = None,
) -> PatchCandidate:
    return PatchCandidate(
        candidate_id=candidate_id,
        agent_id="agent-1",
        patch=StructuredPatch(
            title=f"patch-{candidate_id}",
            summary="candidate summary",
            reasoning="candidate reasoning",
            unified_diff="--- a/app.py\n+++ b/app.py\n@@\n+print('x')\n",
            impacted_files=[PatchTarget(path=f"file-{index}.py") for index in range(impacted_files)],
            risk=risk,
            validation_errors=validation_errors or [],
        ),
    )


def test_judge_returns_structured_scoring() -> None:
    judge = PatchJudge()
    candidate = make_patch_candidate("candidate-1")

    result = judge.evaluate_candidate(candidate)

    assert result.candidate_id == "candidate-1"
    assert result.score.correctness >= 0
    assert result.score.composite_score > 0
    assert result.recommendation in {"accept", "revise", "reject"}
    assert result.critique_summary


def test_judge_selects_best_candidate() -> None:
    judge = PatchJudge()
    collection = CandidateCollection()
    weak = make_patch_candidate("candidate-b", risk=PatchRisk.HIGH, validation_errors=["bad diff"])
    strong = make_patch_candidate("candidate-a", risk=PatchRisk.LOW)
    collection.add_candidate(weak)
    collection.add_candidate(strong)

    winner = judge.select_best_patch(collection)

    assert winner is not None
    assert winner.candidate_id == "candidate-a"
    assert winner.winning_candidate_id == "candidate-a"


def test_judge_handles_ties_deterministically() -> None:
    judge = PatchJudge()
    first = make_patch_candidate("candidate-a")
    second = make_patch_candidate("candidate-b")

    ranked = judge.evaluate_candidates([second, first])

    assert ranked[0].candidate_id == "candidate-a"
    assert ranked[1].candidate_id == "candidate-b"


def test_judge_exposes_critiques() -> None:
    judge = PatchJudge()
    candidate = make_patch_candidate("candidate-1", validation_errors=["syntax error"])

    critiques = judge.critique_summaries([candidate])

    assert "candidate-1" in critiques
    assert "syntax error" in critiques["candidate-1"]
