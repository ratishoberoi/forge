from __future__ import annotations

from backend.runtime.candidate import PatchCandidate
from backend.runtime.patches import PatchTarget, StructuredPatch
from backend.runtime.review_prompting import (
    build_candidate_comparison_prompt,
    build_patch_judge_prompt,
    build_retry_prompt,
)


def make_candidate(candidate_id: str) -> PatchCandidate:
    return PatchCandidate(
        candidate_id=candidate_id,
        agent_id="agent-1",
        patch=StructuredPatch(
            title="candidate patch",
            summary="summary",
            reasoning="reasoning",
            unified_diff="--- a/app.py\n+++ b/app.py\n@@\n+print('x')\n",
            impacted_files=[PatchTarget(path="app.py")],
        ),
    )


def test_patch_judge_prompt_contains_critical_review_guidance() -> None:
    prompt = build_patch_judge_prompt(
        task="fix auth caching",
        candidate=make_candidate("candidate-1"),
        repository_context="app.py handles auth",
    )

    assert "architecture consistency" in prompt
    assert "hallucination risk" in prompt
    assert "candidate-1" in prompt


def test_candidate_comparison_prompt_contains_all_candidates() -> None:
    prompt = build_candidate_comparison_prompt(
        task="fix auth caching",
        candidates=[make_candidate("candidate-1"), make_candidate("candidate-2")],
        repository_context="auth flow",
    )

    assert "candidate-1" in prompt
    assert "candidate-2" in prompt
    assert "prefer the smallest safe patch" in prompt


def test_retry_prompt_contains_critique_and_minimality_guidance() -> None:
    prompt = build_retry_prompt(
        task="fix auth caching",
        critique="Patch touched unrelated files.",
        repository_context="auth flow",
    )

    assert "Patch touched unrelated files." in prompt
    assert "safer, more minimal retry" in prompt
