import pytest
from backend.runtime.convergence_loop import (
    ConvergenceLoop,
    ConvergenceLoopError,
    ConvergenceResult,
)
from backend.runtime.revision_judge import RevisionJudge
from backend.runtime.revision_prompt import RevisionPromptBuilder


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeArtifact:
    def __init__(self, content: str, role: str = "UNKNOWN") -> None:
        self.content = content
        self.role = role
        self.round_id = 1


class FakeCourtroom:
    """Returns fixed artifacts. Tracks call count and received objectives."""

    def __init__(self, critique: str) -> None:
        self.critique = critique
        self.calls = 0
        self.objectives: list[str] = []

    def execute(self, *, objective: str, round_id: int = 1):
        self.calls += 1
        self.objectives.append(objective)
        return [
            FakeArtifact("patch content", role="PRIMARY_CODER"),
            FakeArtifact(self.critique, role="DEEPSEEK_SYNTH"),
            FakeArtifact("judge verdict", role="JUDGE"),
        ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_loop(critique: str) -> tuple[ConvergenceLoop, FakeCourtroom]:
    courtroom = FakeCourtroom(critique=critique)
    loop = ConvergenceLoop(
        courtroom=courtroom,
        judge=RevisionJudge(),
        prompt_builder=RevisionPromptBuilder(),
    )
    return loop, courtroom


# ── RevisionJudge: convergence_score ─────────────────────────────────────────

def test_convergence_score():
    judge = RevisionJudge()

    stable = judge.convergence_score(critique="stable")
    risky = judge.convergence_score(critique="critical bug risk")

    assert stable > risky


def test_convergence_score_clean_critique_is_1():
    judge = RevisionJudge()
    assert judge.convergence_score(critique="looks good") == 1.0


def test_convergence_score_severe_penalty():
    judge = RevisionJudge()
    score = judge.convergence_score(critique="unsafe implementation")
    assert score == 0.6  # 1.0 - 0.4


def test_convergence_score_moderate_penalty():
    judge = RevisionJudge()
    score = judge.convergence_score(critique="there is a risk here")
    assert score == 0.8  # 1.0 - 0.2


def test_convergence_score_multiple_tokens():
    judge = RevisionJudge()
    score = judge.convergence_score(critique="critical failure with a bug")
    # severe: critical(-0.4) + failure(-0.4) + moderate: bug(-0.2) = 1.0 - 1.0 = 0.0
    assert score == 0.0


def test_convergence_score_clamped_to_zero():
    judge = RevisionJudge()
    score = judge.convergence_score(
        critique="unsafe critical failure broken risk issue bug incorrect"
    )
    assert score == 0.0


def test_convergence_score_case_insensitive():
    judge = RevisionJudge()
    score_lower = judge.convergence_score(critique="critical issue")
    score_upper = judge.convergence_score(critique="CRITICAL ISSUE")
    assert score_lower == score_upper


# ── RevisionJudge: should_continue ───────────────────────────────────────────

def test_judge_stops_at_max_iterations():
    judge = RevisionJudge()
    assert judge.should_continue(
        critique="critical bug risk", iteration=3, max_iterations=3
    ) is False


def test_judge_continues_on_unstable():
    judge = RevisionJudge()
    # score = 1.0 - 0.2(risk) - 0.2(bug) = 0.6 < 0.75
    assert judge.should_continue(
        critique="risk and bug found", iteration=1, max_iterations=3
    ) is True


def test_judge_stops_on_clean_critique():
    judge = RevisionJudge()
    # score = 1.0 >= 0.75
    assert judge.should_continue(
        critique="looks good to me", iteration=1, max_iterations=3
    ) is False


def test_judge_custom_threshold():
    judge = RevisionJudge()
    # score = 0.8 (one moderate token)
    assert judge.should_continue(
        critique="minor risk", iteration=1, max_iterations=3, threshold=0.9
    ) is True
    assert judge.should_continue(
        critique="minor risk", iteration=1, max_iterations=3, threshold=0.7
    ) is False


# ── RevisionJudge: verdict ────────────────────────────────────────────────────

def test_judge_verdict_max_iterations():
    judge = RevisionJudge()
    verdict = judge.verdict(
        critique="critical bug", iteration=3, max_iterations=3
    )
    assert "max iterations" in verdict
    assert "STOP" in verdict


def test_judge_verdict_stop_on_stable():
    judge = RevisionJudge()
    verdict = judge.verdict(
        critique="looks good", iteration=1, max_iterations=3
    )
    assert "STOP" in verdict
    assert "convergence score" in verdict


def test_judge_verdict_continue_on_unstable():
    judge = RevisionJudge()
    verdict = judge.verdict(
        critique="critical bug found", iteration=1, max_iterations=3
    )
    assert "CONTINUE" in verdict
    assert "convergence score" in verdict


# ── RevisionJudge: score_breakdown ───────────────────────────────────────────

def test_score_breakdown_populated():
    judge = RevisionJudge()
    breakdown = judge.score_breakdown("critical bug risk")

    assert "critical" in breakdown["matched_severe"]
    assert "bug" in breakdown["matched_moderate"]
    assert "risk" in breakdown["matched_moderate"]
    assert breakdown["score"] < 1.0


def test_score_breakdown_empty_on_clean():
    judge = RevisionJudge()
    breakdown = judge.score_breakdown("looks good")

    assert breakdown["matched_severe"] == []
    assert breakdown["matched_moderate"] == []
    assert breakdown["score"] == 1.0


# ── RevisionPromptBuilder ─────────────────────────────────────────────────────

def test_prompt_builder_basic():
    builder = RevisionPromptBuilder()
    coder = FakeArtifact("def hello(): pass", role="PRIMARY_CODER")
    synth = FakeArtifact("Missing type hints", role="DEEPSEEK_SYNTH")

    prompt = builder.build(
        objective="Add typing",
        coder_artifact=coder,
        synth_artifact=synth,
    )

    assert "OBJECTIVE:" in prompt
    assert "Add typing" in prompt
    assert "PREVIOUS PATCH:" in prompt
    assert "def hello(): pass" in prompt
    assert "ARCHITECTURE CRITIQUE:" in prompt
    assert "Missing type hints" in prompt
    assert "Revise the implementation" in prompt


def test_prompt_builder_with_judge():
    builder = RevisionPromptBuilder()
    coder = FakeArtifact("patch", role="PRIMARY_CODER")
    synth = FakeArtifact("critique", role="DEEPSEEK_SYNTH")
    judge = FakeArtifact("convergence accepted", role="JUDGE")

    prompt = builder.build(
        objective="Add typing",
        coder_artifact=coder,
        synth_artifact=synth,
        judge_artifact=judge,
    )

    assert "JUDGE VERDICT:" in prompt
    assert "convergence accepted" in prompt


def test_prompt_builder_blank_objective_raises():
    builder = RevisionPromptBuilder()
    coder = FakeArtifact("patch")
    synth = FakeArtifact("critique")

    with pytest.raises(ValueError, match="blank"):
        builder.build(objective="   ", coder_artifact=coder, synth_artifact=synth)


def test_prompt_builder_from_history():
    builder = RevisionPromptBuilder()
    history = [[
        FakeArtifact("patch v1", role="PRIMARY_CODER"),
        FakeArtifact("risk detected", role="DEEPSEEK_SYNTH"),
        FakeArtifact("needs revision", role="JUDGE"),
    ]]

    prompt = builder.build_from_history(objective="Add typing", history=history)

    assert "patch v1" in prompt
    assert "risk detected" in prompt
    assert "needs revision" in prompt


def test_prompt_builder_from_empty_history_raises():
    builder = RevisionPromptBuilder()
    with pytest.raises(ValueError, match="empty"):
        builder.build_from_history(objective="Add typing", history=[])


def test_prompt_builder_multi_round_context():
    builder = RevisionPromptBuilder()
    history = [
        [FakeArtifact("patch v1"), FakeArtifact("critique v1")],
        [FakeArtifact("patch v2"), FakeArtifact("critique v2")],
        [FakeArtifact("patch v3"), FakeArtifact("critique v3")],
    ]

    prompt = builder.build_multi_round_context(
        objective="Add typing", history=history, max_rounds=2
    )

    assert "patch v2" in prompt
    assert "patch v3" in prompt
    assert "patch v1" not in prompt


# ── ConvergenceLoop.run ───────────────────────────────────────────────────────

def test_convergence_stops_on_stable():
    # score = 1.0 >= 0.75 — stops immediately
    loop, courtroom = make_loop(critique="looks good no issues")
    history = loop.run(objective="Improve auth", max_iterations=3)

    assert len(history) == 1
    assert courtroom.calls == 1


def test_convergence_retries_on_unstable():
    # score = 1.0 - 0.2(risk) = 0.8... wait, 0.8 >= 0.75 — stops
    # need score < 0.75: use two moderate tokens
    # risk(-0.2) + bug(-0.2) = 0.6 < 0.75 — continues
    loop, courtroom = make_loop(critique="risk and bug detected")
    history = loop.run(objective="Improve auth", max_iterations=3)

    assert len(history) == 3
    assert courtroom.calls == 3


def test_convergence_stops_at_max_even_if_unstable():
    loop, courtroom = make_loop(critique="critical failure bug issue risk")
    history = loop.run(objective="Improve auth", max_iterations=2)

    assert len(history) == 2


def test_convergence_single_iteration():
    loop, courtroom = make_loop(critique="all good")
    history = loop.run(objective="Improve auth", max_iterations=1)

    assert len(history) == 1


def test_convergence_objective_refined_via_prompt_builder():
    loop, courtroom = make_loop(critique="risk and bug detected")
    loop.run(objective="Improve auth", max_iterations=2)

    assert len(courtroom.objectives) == 2
    second = courtroom.objectives[1]
    assert "OBJECTIVE:" in second
    assert "PREVIOUS PATCH:" in second
    assert "ARCHITECTURE CRITIQUE:" in second
    assert "Improve auth" in second


def test_convergence_blank_objective_raises():
    loop, _ = make_loop(critique="stable")
    with pytest.raises(ConvergenceLoopError, match="blank"):
        loop.run(objective="   ", max_iterations=3)


def test_convergence_zero_max_iterations_raises():
    loop, _ = make_loop(critique="stable")
    with pytest.raises(ConvergenceLoopError, match="max_iterations"):
        loop.run(objective="Improve auth", max_iterations=0)


# ── ConvergenceLoop.run_full ──────────────────────────────────────────────────

def test_run_full_returns_convergence_result():
    loop, _ = make_loop(critique="looks good")
    result = loop.run_full(objective="Improve auth", max_iterations=3)
    assert isinstance(result, ConvergenceResult)


def test_run_full_converged_true_on_stable():
    loop, _ = make_loop(critique="looks good no issues")
    result = loop.run_full(objective="Improve auth", max_iterations=3)
    assert result.converged is True
    assert result.iterations_run == 1


def test_run_full_converged_false_on_max_iterations():
    loop, _ = make_loop(critique="risk and bug detected")
    result = loop.run_full(objective="Improve auth", max_iterations=2)
    assert result.converged is False
    assert result.iterations_run == 2


def test_run_full_final_verdict_populated():
    loop, _ = make_loop(critique="looks good")
    result = loop.run_full(objective="Improve auth", max_iterations=3)
    assert result.final_verdict != ""
    assert "convergence score" in result.final_verdict


def test_run_full_all_artifacts_flat():
    loop, _ = make_loop(critique="risk and bug detected")
    result = loop.run_full(objective="Improve auth", max_iterations=2)
    assert len(result.all_artifacts) == 6


def test_run_full_role_filtered_artifacts():
    loop, _ = make_loop(critique="risk and bug detected")
    result = loop.run_full(objective="Improve auth", max_iterations=2)
    assert len(result.coder_artifacts) == 2
    assert len(result.synth_artifacts) == 2
    assert len(result.judge_artifacts) == 2