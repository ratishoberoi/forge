from __future__ import annotations

from backend.runtime.candidate import PatchCandidate
from backend.runtime.judge import JudgeResult, JudgeScore
from backend.runtime.patches import PatchTarget, StructuredPatch
from backend.runtime.retry_prompting import (
    build_convergence_warning_prompt,
    build_retry_candidate_prompt,
    build_self_repair_prompt,
)


def make_candidate() -> PatchCandidate:
    return PatchCandidate(
        candidate_id="candidate-1",
        agent_id="agent-1",
        patch=StructuredPatch(
            title="candidate patch",
            summary="summary",
            reasoning="reasoning",
            unified_diff="--- a/app.py\n+++ b/app.py\n@@\n+print('x')\n",
            impacted_files=[PatchTarget(path="app.py")],
        ),
    )


def make_judge_result() -> JudgeResult:
    return JudgeResult(
        candidate_id="candidate-1",
        score=JudgeScore(7.0, 7.0, 7.0, 7.0, 7.0, 3.0),
        reasoning="Needs a smaller diff.",
        critique_summary="Touched unrelated files.",
        recommendation="revise",
        retry_recommended=True,
        winning_candidate_id="candidate-1",
    )


def test_retry_candidate_prompt_contains_retry_guidance() -> None:
    prompt = build_retry_candidate_prompt(
        task="fix caching",
        candidate=make_candidate(),
        judge_result=make_judge_result(),
        repository_context="auth flow",
    )

    assert "bounded retry" in prompt
    assert "Touched unrelated files." in prompt
    assert "do not hallucinate" in prompt


def test_self_repair_prompt_discourages_rewrites() -> None:
    prompt = build_self_repair_prompt(
        task="fix caching",
        critique="Avoid rewriting the middleware.",
        repository_context="auth flow",
    )

    assert "smallest safe repair" in prompt
    assert "Avoid rewriting the middleware." in prompt
    assert "avoid speculative or hallucinated fixes" in prompt


def test_convergence_warning_prompt_mentions_convergence_reason() -> None:
    prompt = build_convergence_warning_prompt(
        task="fix caching",
        retry_count=2,
        convergence_reason="stagnation_detected",
    )

    assert "stagnation_detected" in prompt
    assert "Retry convergence warning" in prompt
