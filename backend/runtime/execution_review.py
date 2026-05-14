"""Execution evidence normalization for judge and retry layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from backend.runtime.execution import ExecutionResult


class ExecutionSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(slots=True)
class ExecutionReview:
    results: list[ExecutionResult]
    success_rate: float
    severity: ExecutionSeverity
    summary: str
    failure_summary: str
    duration: float
    timeout_count: int
    failure_signature: str | None


def summarize_execution_failures(results: list[ExecutionResult]) -> str:
    failure_lines: list[str] = []
    for result in results:
        if result.success:
            continue
        if result.timed_out:
            failure_lines.append(f"{result.tool} timed out")
            continue
        stderr_line = next(
            (line.strip() for line in result.stderr.splitlines() if line.strip()),
            "",
        )
        stdout_line = next(
            (line.strip() for line in result.stdout.splitlines() if line.strip()),
            "",
        )
        detail = stderr_line or stdout_line or f"exit_code={result.exit_code}"
        failure_lines.append(f"{result.tool} failed: {detail}")
    return "; ".join(failure_lines) if failure_lines else "No execution failures."


def build_execution_summary(results: ExecutionResult | list[ExecutionResult]) -> ExecutionReview:
    normalized_results = results if isinstance(results, list) else [results]
    total = len(normalized_results)
    successes = sum(1 for result in normalized_results if result.success)
    timeout_count = sum(1 for result in normalized_results if result.timed_out)
    duration = round(sum(result.duration for result in normalized_results), 6)
    failure_summary = summarize_execution_failures(normalized_results)

    if timeout_count > 0:
        severity = ExecutionSeverity.CRITICAL
    elif successes < total:
        severity = ExecutionSeverity.WARNING
    else:
        severity = ExecutionSeverity.INFO

    failure_signature = None
    if severity is not ExecutionSeverity.INFO:
        signature_parts = []
        for result in normalized_results:
            if result.success:
                continue
            stderr_compact = " ".join(line.strip() for line in result.stderr.splitlines() if line.strip())
            stdout_compact = " ".join(line.strip() for line in result.stdout.splitlines() if line.strip())
            detail = stderr_compact or stdout_compact
            signature_parts.append(
                f"{result.tool}:{result.exit_code}:{int(result.timed_out)}:{detail[:240]}"
            )
        failure_signature = "|".join(signature_parts) or None

    summary = (
        f"{successes}/{total} execution step(s) succeeded"
        f" in {duration:.3f}s; severity={severity.value}"
    )
    return ExecutionReview(
        results=list(normalized_results),
        success_rate=round(successes / total, 3) if total else 0.0,
        severity=severity,
        summary=summary,
        failure_summary=failure_summary,
        duration=duration,
        timeout_count=timeout_count,
        failure_signature=failure_signature,
    )
