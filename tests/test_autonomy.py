from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.runtime.autonomy import (
    AutonomousLoopController,
    AutonomousSession,
)
from backend.runtime.execution import ExecutionResult
from backend.runtime.judge import JudgeResult, JudgeScore
from backend.runtime.pipeline import PipelineResult
from backend.runtime.pytest_runner import PytestRunner
from backend.runtime.retry import RetryPolicy


@dataclass(slots=True)
class FakePipeline:
    results: list[PipelineResult]
    calls: int = 0

    async def run(self, **_: object) -> PipelineResult:
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


@dataclass(slots=True)
class FakePytestRunner:
    results: list[ExecutionResult]
    calls: int = 0

    async def run(self, cwd: str | Path, *pytest_args: str) -> ExecutionResult:
        assert cwd
        _ = pytest_args
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


class SequenceJudge:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self._index = 0

    def evaluate_candidate(self, candidate, execution_review=None):
        score_value = self._scores[min(self._index, len(self._scores) - 1)]
        self._index += 1
        return JudgeResult(
            candidate_id=candidate.candidate_id,
            score=JudgeScore(
                correctness=score_value,
                architecture=score_value,
                safety=score_value,
                minimality=score_value,
                maintainability=score_value,
                hallucination_risk=max(0.0, 10.0 - score_value),
            ),
            reasoning="sequence judge",
            critique_summary="retry",
            recommendation="revise",
            retry_recommended=True,
            execution_review=execution_review,
        )


def make_pipeline_result(index: int) -> PipelineResult:
    return PipelineResult(
        diff="diff --git a/app.py b/app.py\n@@\n+print('x')\n",
        summary=f"summary-{index}",
        reasoning=f"reasoning-{index}",
        risk="low",
        impacted_files=["app.py"],
        metadata={"agent_id": "coder-agent"},
    )


def make_execution_result(*, success: bool, stderr: str = "", timed_out: bool = False) -> ExecutionResult:
    return ExecutionResult(
        tool="pytest",
        success=success,
        exit_code=0 if success else (None if timed_out else 1),
        stdout="1 passed" if success else "",
        stderr=stderr if stderr else ("Execution timed out after 1 seconds." if timed_out else "Traceback\nAssertionError"),
        duration=0.1,
        timed_out=timed_out,
    )


def make_session(tmp_path: Path, *, max_iterations: int = 4, cancellation_event: asyncio.Event | None = None) -> AutonomousSession:
    return AutonomousSession(
        task="fix auth caching",
        repository_context="auth flow",
        impacted_files=["app.py"],
        workspace=tmp_path,
        max_iterations=max_iterations,
        cancellation_event=cancellation_event,
    )


@pytest.mark.asyncio
async def test_successful_convergence(tmp_path: Path) -> None:
    controller = AutonomousLoopController(
        pipeline=FakePipeline([make_pipeline_result(1)]),
        pytest_runner=FakePytestRunner([make_execution_result(success=True)]),  # type: ignore[arg-type]
    )

    result = await controller.run_until_converged(make_session(tmp_path))

    assert result.stop_reason == "accepted"
    assert result.converged is True
    assert result.best_candidate is not None


@pytest.mark.asyncio
async def test_retry_refinement_then_accept(tmp_path: Path) -> None:
    controller = AutonomousLoopController(
        pipeline=FakePipeline([make_pipeline_result(1), make_pipeline_result(2)]),
        pytest_runner=FakePytestRunner([
            make_execution_result(success=False, stderr="Traceback\nAssertionError: broken"),
            make_execution_result(success=True),
        ]),  # type: ignore[arg-type]
    )

    result = await controller.run_until_converged(make_session(tmp_path, max_iterations=3))

    assert len(result.iterations) == 2
    assert result.stop_reason == "accepted"
    assert result.best_candidate is not None
    assert result.retry_history


@pytest.mark.asyncio
async def test_stagnation_stop(tmp_path: Path) -> None:
    controller = AutonomousLoopController(
        pipeline=FakePipeline([make_pipeline_result(1), make_pipeline_result(2)]),
        pytest_runner=FakePytestRunner([
            make_execution_result(success=False, stderr="Traceback\nAssertionError: first"),
            make_execution_result(success=False, stderr="Traceback\nAssertionError: second"),
        ]),  # type: ignore[arg-type]
    )

    result = await controller.run_until_converged(make_session(tmp_path, max_iterations=4))

    assert result.stop_reason == "stagnation"
    assert result.converged is False


@pytest.mark.asyncio
async def test_oscillation_stop(tmp_path: Path) -> None:
    controller = AutonomousLoopController(
        pipeline=FakePipeline([
            make_pipeline_result(1),
            make_pipeline_result(2),
            make_pipeline_result(3),
            make_pipeline_result(4),
        ]),
        pytest_runner=FakePytestRunner([make_execution_result(success=True)] * 4),  # type: ignore[arg-type]
        judge=SequenceJudge([6.0, 7.0, 6.0, 7.0]),  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_retries=4),
    )

    result = await controller.run_until_converged(make_session(tmp_path, max_iterations=5))

    assert result.stop_reason == "oscillation"


@pytest.mark.asyncio
async def test_repeated_execution_failure_stop(tmp_path: Path) -> None:
    failure = make_execution_result(success=False, stderr="Traceback\nAssertionError: same failure")
    controller = AutonomousLoopController(
        pipeline=FakePipeline([make_pipeline_result(1), make_pipeline_result(2)]),
        pytest_runner=FakePytestRunner([failure, failure]),  # type: ignore[arg-type]
    )

    result = await controller.run_until_converged(make_session(tmp_path, max_iterations=4))

    assert result.stop_reason == "execution_failure"
    assert len(result.execution_failure_history) >= 2


@pytest.mark.asyncio
async def test_best_candidate_retention(tmp_path: Path) -> None:
    controller = AutonomousLoopController(
        pipeline=FakePipeline([make_pipeline_result(1), make_pipeline_result(2)]),
        pytest_runner=FakePytestRunner([
            make_execution_result(success=False, stderr="Traceback\nAssertionError: broken"),
            make_execution_result(success=True),
        ]),  # type: ignore[arg-type]
    )

    result = await controller.run_until_converged(make_session(tmp_path, max_iterations=3))

    assert result.best_candidate is not None
    assert result.final_candidate is not None
    assert result.best_candidate.score >= result.final_candidate.score


@pytest.mark.asyncio
async def test_bounded_iteration_guarantees(tmp_path: Path) -> None:
    controller = AutonomousLoopController(
        pipeline=FakePipeline([make_pipeline_result(1), make_pipeline_result(2), make_pipeline_result(3)]),
        pytest_runner=FakePytestRunner([
            make_execution_result(success=False, stderr="Traceback\nAssertionError: one"),
            make_execution_result(success=False, stderr="Traceback\nAssertionError: two"),
            make_execution_result(success=False, stderr="Traceback\nAssertionError: three"),
        ]),  # type: ignore[arg-type]
    )

    result = await controller.run_until_converged(make_session(tmp_path, max_iterations=1))

    assert len(result.iterations) == 1
    assert result.stop_reason == "max_iterations"


@pytest.mark.asyncio
async def test_cancellation_behavior(tmp_path: Path) -> None:
    cancellation_event = asyncio.Event()

    class CancellingPipeline(FakePipeline):
        async def run(self, **kwargs: object) -> PipelineResult:
            result = await super().run(**kwargs)
            cancellation_event.set()
            return result

    controller = AutonomousLoopController(
        pipeline=CancellingPipeline([make_pipeline_result(1), make_pipeline_result(2)]),
        pytest_runner=FakePytestRunner([
            make_execution_result(success=False, stderr="Traceback\nAssertionError: one"),
            make_execution_result(success=True),
        ]),  # type: ignore[arg-type]
    )

    result = await controller.run_until_converged(
        make_session(tmp_path, max_iterations=3, cancellation_event=cancellation_event)
    )

    assert result.stop_reason == "cancelled"
    assert result.cancelled is True


@pytest.mark.asyncio
async def test_deterministic_stopping(tmp_path: Path) -> None:
    controller = AutonomousLoopController(
        pipeline=FakePipeline([make_pipeline_result(1), make_pipeline_result(2)]),
        pytest_runner=FakePytestRunner([
            make_execution_result(success=False, stderr="Traceback\nAssertionError: same"),
            make_execution_result(success=False, stderr="Traceback\nAssertionError: same"),
        ]),  # type: ignore[arg-type]
    )

    first = await controller.run_until_converged(make_session(tmp_path, max_iterations=4))
    second = await controller.run_until_converged(make_session(tmp_path, max_iterations=4))

    assert first.stop_reason == second.stop_reason == "execution_failure"
