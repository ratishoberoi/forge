"""Judge-driven retry and self-repair orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.runtime.candidate import PatchCandidate
from backend.runtime.convergence import ConvergenceTracker
from backend.runtime.execution_review import ExecutionSeverity
from backend.runtime.judge import JudgeResult
from backend.runtime.retry_prompting import (
    build_convergence_warning_prompt,
    build_retry_candidate_prompt,
)


@dataclass(slots=True)
class RetryPolicy:
    max_retries: int = 2
    min_score_improvement: float = 0.25
    max_stagnant_rounds: int = 2
    acceptance_score: float = 8.0
    score_tolerance: float = 0.0


@dataclass(slots=True)
class RetryDecision:
    should_retry: bool
    retry_index: int
    reason: str
    retry_prompt: str | None = None
    convergence_warning: str | None = None
    stop_reason: str | None = None
    execution_aware: bool = False
    execution_failure_signature: str | None = None
    retry_priority: str = "normal"


@dataclass(slots=True)
class RetryResult:
    final_candidate: PatchCandidate
    best_candidate: PatchCandidate
    retry_count: int
    history: list[str] = field(default_factory=list)
    score_history: list[float] = field(default_factory=list)
    converged: bool = False
    stagnated: bool = False
    stop_reason: str | None = None
    execution_failure_history: list[str] = field(default_factory=list)
    escalated: bool = False


class RetryOrchestrator:
    """Deterministic retry policy over judged patch candidates."""

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        self.policy = policy or RetryPolicy()
        self._tracker = ConvergenceTracker()
        self._history: list[PatchCandidate] = []
        self._best_candidate: PatchCandidate | None = None
        self._execution_failure_history: list[str] = []

    def decide_retry(
        self,
        *,
        task: str,
        candidate: PatchCandidate,
        judge_result: JudgeResult,
        repository_context: str,
    ) -> RetryDecision:
        self._register_candidate(candidate, judge_result)
        retry_index = max(0, len(self._history) - 1)
        execution_review = judge_result.execution_review
        failure_signature = execution_review.failure_signature if execution_review is not None else None
        execution_aware = execution_review is not None
        retry_priority = judge_result.retry_priority

        if not judge_result.retry_recommended or judge_result.score.composite_score >= self.policy.acceptance_score:
            return RetryDecision(
                should_retry=False,
                retry_index=retry_index,
                reason="judge_accepted_candidate",
                stop_reason="accepted",
                execution_aware=execution_aware,
                execution_failure_signature=failure_signature,
                retry_priority=retry_priority,
            )

        if self._has_repeated_execution_failure(failure_signature):
            return RetryDecision(
                should_retry=False,
                retry_index=retry_index,
                reason="repeated_execution_failure",
                stop_reason="repeated_execution_failure",
                execution_aware=execution_aware,
                execution_failure_signature=failure_signature,
                retry_priority="high",
            )

        converged, convergence_reason = self._tracker.convergence_status(
            min_improvement=self.policy.min_score_improvement,
            max_stagnant_rounds=self.policy.max_stagnant_rounds,
            tolerance=self.policy.score_tolerance,
        )
        if converged:
            return RetryDecision(
                should_retry=False,
                retry_index=retry_index,
                reason=convergence_reason or "converged",
                convergence_warning=build_convergence_warning_prompt(
                    task=task,
                    retry_count=retry_index,
                    convergence_reason=convergence_reason or "converged",
                ),
                stop_reason=convergence_reason or "converged",
                execution_aware=execution_aware,
                execution_failure_signature=failure_signature,
                retry_priority=retry_priority,
            )

        if retry_index >= self.policy.max_retries:
            return RetryDecision(
                should_retry=False,
                retry_index=retry_index,
                reason="retry_limit_reached",
                stop_reason="retry_limit_reached",
                execution_aware=execution_aware,
                execution_failure_signature=failure_signature,
                retry_priority=retry_priority,
            )

        if execution_review is not None and execution_review.severity is ExecutionSeverity.CRITICAL:
            reason = "critical_execution_failure"
        else:
            reason = "judge_requested_retry"

        return RetryDecision(
            should_retry=True,
            retry_index=retry_index + 1,
            reason=reason,
            retry_prompt=build_retry_candidate_prompt(
                task=task,
                candidate=candidate,
                judge_result=judge_result,
                repository_context=repository_context,
            ),
            execution_aware=execution_aware,
            execution_failure_signature=failure_signature,
            retry_priority="high" if reason == "critical_execution_failure" else retry_priority,
        )

    def finalize(self) -> RetryResult:
        if not self._history or self._best_candidate is None:
            raise ValueError("Cannot finalize retry orchestration without any candidate history.")

        last_candidate = self._history[-1]
        converged, convergence_reason = self._tracker.convergence_status(
            min_improvement=self.policy.min_score_improvement,
            max_stagnant_rounds=self.policy.max_stagnant_rounds,
            tolerance=self.policy.score_tolerance,
        )
        return RetryResult(
            final_candidate=last_candidate,
            best_candidate=self._best_candidate,
            retry_count=max(0, len(self._history) - 1),
            history=[candidate.candidate_id for candidate in self._history],
            score_history=list(self._tracker.score_history),
            converged=converged,
            stagnated=convergence_reason == "stagnation_detected",
            stop_reason=convergence_reason,
            execution_failure_history=list(self._execution_failure_history),
            escalated=any(candidate.judge_result and candidate.judge_result.retry_priority == "high" for candidate in self._history),
        )

    def history(self) -> list[PatchCandidate]:
        return list(self._history)

    def _register_candidate(self, candidate: PatchCandidate, judge_result: JudgeResult) -> None:
        candidate.judge_result = judge_result
        candidate.critique = judge_result.critique_summary
        self._history.append(candidate)
        self._tracker.record(
            judge_result.score.composite_score,
            min_improvement=self.policy.min_score_improvement,
        )
        if judge_result.execution_review is not None and judge_result.execution_review.failure_signature is not None:
            self._execution_failure_history.append(judge_result.execution_review.failure_signature)
        if self._best_candidate is None:
            self._best_candidate = candidate
            return
        current_best_score = self._best_candidate.score if self._best_candidate.score is not None else float("-inf")
        candidate_score = candidate.score if candidate.score is not None else float("-inf")
        if candidate_score > current_best_score:
            self._best_candidate = candidate

    def _has_repeated_execution_failure(self, failure_signature: str | None) -> bool:
        if failure_signature is None:
            return False
        if len(self._execution_failure_history) < 2:
            return False
        return self._execution_failure_history[-1] == self._execution_failure_history[-2] == failure_signature
