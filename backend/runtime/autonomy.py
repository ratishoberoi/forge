"""Bounded autonomous cognition orchestration loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from backend.runtime.candidate import PatchCandidate
from backend.runtime.execution import ExecutionResult
from backend.runtime.execution_review import ExecutionReview, build_execution_summary
from backend.runtime.judge import JudgeResult, PatchJudge
from backend.runtime.patches import PatchRisk, PatchTarget, StructuredPatch
from backend.runtime.pipeline import AutonomousExecutionPipeline, PipelineResult
from backend.runtime.pytest_runner import PytestRunner
from backend.runtime.retry import RetryDecision, RetryOrchestrator, RetryPolicy
from backend.runtime.validation import PatchValidator


@dataclass(slots=True)
class AutonomousIteration:
    index: int
    task_prompt: str
    candidate: PatchCandidate | None = None
    judge_result: JudgeResult | None = None
    execution_result: ExecutionResult | None = None
    execution_review: ExecutionReview | None = None
    retry_decision: RetryDecision | None = None
    error: str | None = None


@dataclass(slots=True)
class AutonomousSession:
    task: str
    repository_context: str
    impacted_files: list[str]
    workspace: str | Path
    pytest_args: tuple[str, ...] = ()
    agent_id: str = "coder-agent"
    max_iterations: int = 3
    cancellation_event: asyncio.Event | None = None


@dataclass(slots=True)
class AutonomousLoopResult:
    session: AutonomousSession
    iterations: list[AutonomousIteration] = field(default_factory=list)
    final_candidate: PatchCandidate | None = None
    best_candidate: PatchCandidate | None = None
    stop_reason: str = "rejected"
    converged: bool = False
    cancelled: bool = False
    score_history: list[float] = field(default_factory=list)
    retry_history: list[str] = field(default_factory=list)
    execution_failure_history: list[str] = field(default_factory=list)


class AutonomousLoopController:
    """Deterministic generate/execute/judge/retry loop."""

    def __init__(
        self,
        *,
        pipeline: AutonomousExecutionPipeline,
        pytest_runner: PytestRunner,
        judge: PatchJudge | None = None,
        retry_policy: RetryPolicy | None = None,
        validator: PatchValidator | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.pytest_runner = pytest_runner
        self.judge = judge or PatchJudge()
        self.retry_policy = retry_policy or RetryPolicy()
        self.validator = validator or PatchValidator()

    async def run_iteration(
        self,
        session: AutonomousSession,
        *,
        iteration_index: int,
        task_prompt: str,
        retry_orchestrator: RetryOrchestrator | None = None,
    ) -> AutonomousIteration:
        pipeline_result = await self.pipeline.run(
            task=task_prompt,
            repository_context=session.repository_context,
            impacted_files=session.impacted_files,
            agent_id=session.agent_id,
        )
        candidate = self._candidate_from_pipeline_result(
            session=session,
            pipeline_result=pipeline_result,
        )
        execution_result = await self.pytest_runner.run(
            session.workspace,
            *session.pytest_args,
        )
        execution_review = build_execution_summary(execution_result)
        judge_result = self.judge.evaluate_candidate(candidate, execution_review)
        retry_decision = None
        if retry_orchestrator is not None:
            retry_decision = retry_orchestrator.decide_retry(
                task=session.task,
                candidate=candidate,
                judge_result=judge_result,
                repository_context=session.repository_context,
            )
        return AutonomousIteration(
            index=iteration_index,
            task_prompt=task_prompt,
            candidate=candidate,
            judge_result=judge_result,
            execution_result=execution_result,
            execution_review=execution_review,
            retry_decision=retry_decision,
        )

    async def run_until_converged(self, session: AutonomousSession) -> AutonomousLoopResult:
        retry_orchestrator = RetryOrchestrator(self.retry_policy)
        iterations: list[AutonomousIteration] = []
        task_prompt = session.task

        try:
            for iteration_index in range(session.max_iterations):
                if self._is_cancelled(session):
                    return self._build_result(
                        session=session,
                        iterations=iterations,
                        retry_orchestrator=retry_orchestrator,
                        stop_reason="cancelled",
                        converged=False,
                        cancelled=True,
                    )

                iteration = await self.run_iteration(
                    session,
                    iteration_index=iteration_index,
                    task_prompt=task_prompt,
                    retry_orchestrator=retry_orchestrator,
                )
                iterations.append(iteration)

                if iteration.retry_decision is None:
                    return self._build_result(
                        session=session,
                        iterations=iterations,
                        retry_orchestrator=retry_orchestrator,
                        stop_reason="rejected",
                        converged=False,
                    )

                if not iteration.retry_decision.should_retry:
                    return self._build_result(
                        session=session,
                        iterations=iterations,
                        retry_orchestrator=retry_orchestrator,
                        stop_reason=self._classify_stop_reason(iteration.retry_decision.stop_reason, iteration),
                        converged=iteration.retry_decision.stop_reason == "accepted",
                    )

                if iteration_index + 1 >= session.max_iterations:
                    return self._build_result(
                        session=session,
                        iterations=iterations,
                        retry_orchestrator=retry_orchestrator,
                        stop_reason="max_iterations",
                        converged=False,
                    )

                task_prompt = iteration.retry_decision.retry_prompt or task_prompt
        except asyncio.CancelledError:
            return self._build_result(
                session=session,
                iterations=iterations,
                retry_orchestrator=retry_orchestrator,
                stop_reason="cancelled",
                converged=False,
                cancelled=True,
            )
        except Exception as exc:
            iterations.append(
                AutonomousIteration(
                    index=len(iterations),
                    task_prompt=task_prompt,
                    error=str(exc),
                )
            )
            return self._build_result(
                session=session,
                iterations=iterations,
                retry_orchestrator=retry_orchestrator,
                stop_reason="rejected",
                converged=False,
            )

        return self._build_result(
            session=session,
            iterations=iterations,
            retry_orchestrator=retry_orchestrator,
            stop_reason="max_iterations",
            converged=False,
        )

    def _candidate_from_pipeline_result(
        self,
        *,
        session: AutonomousSession,
        pipeline_result: PipelineResult,
    ) -> PatchCandidate:
        patch = StructuredPatch(
            title=session.task,
            description=pipeline_result.reasoning,
            unified_diff=pipeline_result.diff,
            impacted_files=[PatchTarget(path=path) for path in pipeline_result.impacted_files],
            risk=PatchRisk(pipeline_result.risk),
            metadata=dict(pipeline_result.metadata),
            summary=pipeline_result.summary,
            reasoning=pipeline_result.reasoning,
        )
        validated = self.validator.validate(patch)
        return PatchCandidate(
            patch=validated,
            agent_id=str(pipeline_result.metadata.get("agent_id", session.agent_id)),
        )

    def _build_result(
        self,
        *,
        session: AutonomousSession,
        iterations: list[AutonomousIteration],
        retry_orchestrator: RetryOrchestrator,
        stop_reason: str,
        converged: bool,
        cancelled: bool = False,
    ) -> AutonomousLoopResult:
        best_candidate = None
        final_candidate = None
        score_history: list[float] = []
        retry_history: list[str] = []
        execution_failure_history: list[str] = []

        if retry_orchestrator.history():
            retry_result = retry_orchestrator.finalize()
            best_candidate = retry_result.best_candidate
            final_candidate = retry_result.final_candidate
            score_history = retry_result.score_history
            retry_history = retry_result.history
            execution_failure_history = retry_result.execution_failure_history

        return AutonomousLoopResult(
            session=session,
            iterations=iterations,
            final_candidate=final_candidate,
            best_candidate=best_candidate,
            stop_reason=stop_reason,
            converged=converged,
            cancelled=cancelled,
            score_history=score_history,
            retry_history=retry_history,
            execution_failure_history=execution_failure_history,
        )

    @staticmethod
    def _classify_stop_reason(raw_reason: str | None, iteration: AutonomousIteration) -> str:
        if raw_reason == "accepted":
            return "accepted"
        if raw_reason in {"stagnation_detected", "repeated_score_detected"}:
            return "stagnation"
        if raw_reason == "oscillation_detected":
            return "oscillation"
        if raw_reason == "repeated_execution_failure":
            return "execution_failure"
        if raw_reason == "retry_limit_reached":
            if (
                iteration.execution_review is not None
                and iteration.execution_review.success_rate == 0.0
            ):
                return "execution_failure"
            if iteration.judge_result is not None and iteration.judge_result.recommendation == "reject":
                return "rejected"
            return "max_iterations"
        if raw_reason == "critical_execution_failure":
            return "execution_failure"
        return "rejected"

    @staticmethod
    def _is_cancelled(session: AutonomousSession) -> bool:
        return session.cancellation_event.is_set() if session.cancellation_event is not None else False
