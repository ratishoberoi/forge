from __future__ import annotations

from backend.runtime.execution import ExecutionResult
from backend.runtime.execution_review import (
    ExecutionSeverity,
    build_execution_summary,
    summarize_execution_failures,
)


def test_execution_summary_classifies_success_as_info() -> None:
    review = build_execution_summary(
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

    assert review.severity is ExecutionSeverity.INFO
    assert review.success_rate == 1.0


def test_execution_summary_classifies_timeout_as_critical() -> None:
    review = build_execution_summary(
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

    assert review.severity is ExecutionSeverity.CRITICAL
    assert review.timeout_count == 1
    assert review.failure_signature is not None


def test_summarize_execution_failures_uses_stderr_evidence() -> None:
    summary = summarize_execution_failures(
        [
            ExecutionResult(
                tool="pytest",
                success=False,
                exit_code=1,
                stdout="",
                stderr="Traceback\\nAssertionError: boom",
                duration=0.1,
                timed_out=False,
            )
        ]
    )

    assert "pytest failed" in summary
    assert "Traceback" in summary
