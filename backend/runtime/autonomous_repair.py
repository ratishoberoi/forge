from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from backend.runtime.execution_result import ExecutionResult
from backend.runtime.execution_runner import ExecutionRunner, ExecutionRunnerError
from backend.runtime.patch_writer import PatchResult
from backend.runtime.repository_execution_engine import (
    RepositoryExecutionApplyResult,
    RepositoryExecutionEngine,
    RepositoryExecutionPreparation,
    RepositoryExecutionError,
)


class FailureType(StrEnum):
    SYNTAX_ERROR = "SyntaxError"
    IMPORT_ERROR = "ImportError"
    MODULE_NOT_FOUND = "ModuleNotFoundError"
    ASSERTION_ERROR = "AssertionError"
    TYPE_ERROR = "TypeError"
    RUNTIME_ERROR = "RuntimeError"
    TEST_FAILURE = "Test Failure"
    BUILD_FAILURE = "Build Failure"
    LINT_FAILURE = "Lint Failure"
    UNKNOWN = "Unknown"


@dataclass(slots=True)
class FailureAnalysis:
    category: FailureType
    file: str | None
    traceback: str
    failing_tests: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "file": self.file,
            "traceback": self.traceback,
            "failing_tests": self.failing_tests,
            "message": self.message,
        }


@dataclass(slots=True)
class TestExecutionResult:
    passed: bool
    failed: bool
    return_code: int
    stdout: str
    stderr: str
    failing_tests: list[str]
    command: list[str]
    duration_seconds: float
    failure: FailureAnalysis | None = None

    @classmethod
    def from_execution(
        cls,
        result: ExecutionResult,
        failure: FailureAnalysis | None,
    ) -> TestExecutionResult:
        return cls(
            passed=result.succeeded,
            failed=result.failed,
            return_code=result.return_code,
            stdout=result.stdout,
            stderr=result.stderr,
            failing_tests=failure.failing_tests if failure else [],
            command=result.command,
            duration_seconds=result.duration_seconds,
            failure=failure,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "failing_tests": self.failing_tests,
            "command": self.command,
            "duration_seconds": self.duration_seconds,
            "failure": self.failure.to_dict() if self.failure else None,
        }


@dataclass(slots=True)
class RepairContext:
    objective: str
    repair_objective: str
    failure: FailureAnalysis
    failing_file: str | None
    failing_file_content: str
    failing_test: str | None
    failing_test_content: str
    last_coder_artifact: str
    last_synth_artifact: str
    repository_context: dict[str, Any]

    def to_prompt(self) -> str:
        return (
            "REPAIR OBJECTIVE\n"
            f"{self.repair_objective}\n\n"
            "Return strict PRIMARY_CODER JSON only: "
            '{"summary":"...","files":{"path":"full content"}}.\n\n'
            f"REPAIR_CONTEXT:\n{json.dumps(self.to_dict(), indent=2)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "repair_objective": self.repair_objective,
            "failure": self.failure.to_dict(),
            "failing_file": self.failing_file,
            "failing_file_content": self.failing_file_content,
            "failing_test": self.failing_test,
            "failing_test_content": self.failing_test_content,
            "last_coder_artifact": self.last_coder_artifact,
            "last_synth_artifact": self.last_synth_artifact,
            "repository_context": self.repository_context,
        }


@dataclass(slots=True)
class RepairHistoryEntry:
    attempt: int
    phase: str
    failure_category: str | None = None
    failing_test: str | None = None
    patch_success: bool | None = None
    tests_passed: bool | None = None
    test_pass_rate: float = 0.0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "phase": self.phase,
            "failure_category": self.failure_category,
            "failing_test": self.failing_test,
            "patch_success": self.patch_success,
            "tests_passed": self.tests_passed,
            "test_pass_rate": self.test_pass_rate,
            "message": self.message,
        }


@dataclass(slots=True)
class ConvergenceState:
    iteration_count: int = 0
    repair_count: int = 0
    success: bool = False
    current_phase: str = "idle"
    status: str = "running"
    stop_reason: str | None = None
    last_failure_type: FailureType | None = None
    last_failing_test: str | None = None
    test_pass_rate: float = 0.0
    repair_limit: int = 0
    history: list[RepairHistoryEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_count": self.iteration_count,
            "repair_count": self.repair_count,
            "success": self.success,
            "current_phase": self.current_phase,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "last_failure_type": self.last_failure_type.value if self.last_failure_type else None,
            "last_failing_test": self.last_failing_test,
            "test_pass_rate": self.test_pass_rate,
            "repair_limit": self.repair_limit,
            "history": [entry.to_dict() for entry in self.history],
        }


@dataclass(slots=True)
class RepairConvergenceResult:
    state: ConvergenceState
    initial_patch: RepositoryExecutionApplyResult | None
    final_patch: RepositoryExecutionApplyResult | None
    final_execution: TestExecutionResult | None
    repair_contexts: list[RepairContext] = field(default_factory=list)
    telemetry: list[str] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.state.success

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "state": self.state.to_dict(),
            "initial_patch": self.initial_patch.to_dict() if self.initial_patch else None,
            "final_patch": self.final_patch.to_dict() if self.final_patch else None,
            "final_execution": self.final_execution.to_dict() if self.final_execution else None,
            "repair_contexts": [context.to_dict() for context in self.repair_contexts],
            "telemetry": self.telemetry,
        }


RepairGenerator = Callable[[RepairContext], str]


class FailureClassifier:
    TRACEBACK_LIMIT = 6000

    def classify(self, result: ExecutionResult) -> FailureAnalysis | None:
        if result.succeeded:
            return None

        output = f"{result.stdout}\n{result.stderr}".strip()
        category = self._category(result.command, output)
        failing_tests = self._failing_tests(output)
        file_path = self._file(output, failing_tests)
        traceback = output[-self.TRACEBACK_LIMIT:]
        message = self._message(output, category)
        return FailureAnalysis(
            category=category,
            file=file_path,
            traceback=traceback,
            failing_tests=failing_tests,
            message=message,
        )

    @staticmethod
    def _category(command: list[str], output: str) -> FailureType:
        for failure_type in (
            FailureType.MODULE_NOT_FOUND,
            FailureType.IMPORT_ERROR,
            FailureType.SYNTAX_ERROR,
            FailureType.ASSERTION_ERROR,
            FailureType.TYPE_ERROR,
            FailureType.RUNTIME_ERROR,
        ):
            if failure_type.value in output:
                return failure_type

        command_text = " ".join(command).lower()
        if any(tool in command_text for tool in ("ruff", "flake8", "eslint", "mypy", "lint")):
            return FailureType.LINT_FAILURE
        if any(tool in command_text for tool in ("build", "compile", "tsc")):
            return FailureType.BUILD_FAILURE
        if "failed" in output.lower() or "error" in output.lower():
            return FailureType.TEST_FAILURE
        return FailureType.UNKNOWN

    @staticmethod
    def _failing_tests(output: str) -> list[str]:
        tests: list[str] = []
        patterns = (
            r"^(?:FAILED|ERROR)\s+([^\s]+::[^\s]+)",
            r"^([A-Za-z0-9_./\\-]+\.py::[A-Za-z0-9_:\[\].-]+)",
            r"^FAIL\s+([^\s]+)",
        )
        for line in output.splitlines():
            stripped = line.strip()
            for pattern in patterns:
                match = re.search(pattern, stripped)
                if match:
                    tests.append(match.group(1).replace("\\", "/"))
                    break
        return sorted(dict.fromkeys(tests))

    @staticmethod
    def _file(output: str, failing_tests: list[str]) -> str | None:
        if failing_tests:
            first = failing_tests[0].split("::", 1)[0]
            if first:
                return first
        traceback_matches = re.findall(r'File "([^"]+)", line \d+', output)
        if traceback_matches:
            return traceback_matches[-1].replace("\\", "/")
        location = re.search(r"([A-Za-z0-9_./\\-]+\.(?:py|ts|tsx|js|jsx)):\d+", output)
        if location:
            return location.group(1).replace("\\", "/")
        return None

    @staticmethod
    def _message(output: str, category: FailureType) -> str:
        for line in reversed(output.splitlines()):
            stripped = line.strip()
            if stripped and (category.value in stripped or "Error" in stripped or "failed" in stripped):
                return stripped[:500]
        return output.splitlines()[-1][:500] if output.splitlines() else category.value


class RepairContextBuilder:
    MAX_CONTENT_CHARS = 8000

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()

    def build(
        self,
        *,
        objective: str,
        preparation: RepositoryExecutionPreparation,
        failure: FailureAnalysis,
        last_coder_artifact: str,
        last_synth_artifact: str = "",
    ) -> RepairContext:
        failing_test = failure.failing_tests[0] if failure.failing_tests else None
        failing_file = self._relative_to_repo(failure.file)
        test_path = self._relative_to_repo(failing_test.split("::", 1)[0]) if failing_test else None
        repository_context = preparation.context.to_dict()
        if failing_file and failing_file not in repository_context["file_summaries"]:
            content = self._read_limited(failing_file)
            if content:
                repository_context["file_summaries"][failing_file] = content
                repository_context["relevant_files"] = sorted(
                    set(repository_context["relevant_files"]) | {failing_file}
                )

        repair_objective = (
            f"Repair failing tests for objective: {objective}. "
            f"Failure category: {failure.category.value}. "
            "Patch only the files needed to make validation pass."
        )
        return RepairContext(
            objective=objective,
            repair_objective=repair_objective,
            failure=failure,
            failing_file=failing_file,
            failing_file_content=self._read_limited(failing_file),
            failing_test=failing_test,
            failing_test_content=self._read_limited(test_path),
            last_coder_artifact=last_coder_artifact[-self.MAX_CONTENT_CHARS:],
            last_synth_artifact=last_synth_artifact[-self.MAX_CONTENT_CHARS:],
            repository_context=repository_context,
        )

    def _relative_to_repo(self, path: str | None) -> str | None:
        if not path:
            return None
        raw_path = path.split("::", 1)[0]
        candidate = Path(raw_path)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(self.repo_root).as_posix()
            except ValueError:
                return None
        normalized = Path(raw_path.replace("\\", "/"))
        if ".." in normalized.parts:
            return None
        return normalized.as_posix()

    def _read_limited(self, path: str | None) -> str:
        if not path:
            return ""
        target = (self.repo_root / path).resolve()
        if not target.is_relative_to(self.repo_root) or not target.is_file():
            return ""
        try:
            return target.read_text(encoding="utf-8", errors="replace")[: self.MAX_CONTENT_CHARS]
        except OSError:
            return ""


class AutonomousRepairConvergenceEngine:
    """Closed-loop patch, test, failure analysis, and repair orchestration."""

    def __init__(
        self,
        *,
        repo_root: str,
        repository_engine: RepositoryExecutionEngine | None = None,
        runner: ExecutionRunner | None = None,
        classifier: FailureClassifier | None = None,
        repair_generator: RepairGenerator | None = None,
        telemetry: Callable[[str], None] | None = None,
        runtime_budget_seconds: float = 900.0,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.repository_engine = repository_engine or RepositoryExecutionEngine(repo_root=str(self.repo_root))
        self.runner = runner or ExecutionRunner()
        self.classifier = classifier or FailureClassifier()
        self.repair_generator = repair_generator
        self.context_builder = RepairContextBuilder(str(self.repo_root))
        self.runtime_budget_seconds = runtime_budget_seconds
        self._telemetry_sink = telemetry
        self._telemetry: list[str] = []

    def run(
        self,
        *,
        objective: str,
        preparation: RepositoryExecutionPreparation,
        initial_response_text: str,
        test_command: list[str],
        validation_commands: list[list[str]] | None = None,
        max_repairs: int = 2,
        test_env: dict[str, str] | None = None,
        last_synth_artifact: str = "",
    ) -> RepairConvergenceResult:
        if max_repairs < 0:
            raise ValueError("max_repairs must be >= 0.")
        if not test_command:
            raise ValueError("test_command must not be empty.")

        state = ConvergenceState(repair_limit=max_repairs)
        started_at = time.monotonic()
        initial_patch = self._apply_patch(
            response_text=initial_response_text,
            preparation=preparation,
            state=state,
            attempt=0,
        )
        final_patch = initial_patch
        if not initial_patch.success:
            state.status = "failed"
            state.stop_reason = "patch_apply_failed"
            state.current_phase = "failed"
            return RepairConvergenceResult(
                state=state,
                initial_patch=initial_patch,
                final_patch=final_patch,
                final_execution=None,
                telemetry=self._telemetry,
            )

        final_execution = self._run_tests(
            command=test_command,
            validation_commands=validation_commands,
            test_env=test_env,
            state=state,
            attempt=0,
        )
        if final_execution.passed:
            self._mark_success(state)
            return RepairConvergenceResult(
                state=state,
                initial_patch=initial_patch,
                final_patch=final_patch,
                final_execution=final_execution,
                telemetry=self._telemetry,
            )

        repair_contexts: list[RepairContext] = []
        last_coder_artifact = initial_response_text
        while state.repair_count < max_repairs:
            if time.monotonic() - started_at > self.runtime_budget_seconds:
                state.status = "timeout"
                state.stop_reason = "runtime_budget_exceeded"
                state.current_phase = "timeout"
                break
            if final_execution.failure is None:
                state.status = "failed"
                state.stop_reason = "failure_analysis_unavailable"
                state.current_phase = "failed"
                break
            if self.repair_generator is None:
                state.status = "failed"
                state.stop_reason = "repair_generator_unavailable"
                state.current_phase = "failed"
                break

            state.repair_count += 1
            attempt = state.repair_count
            context = self.context_builder.build(
                objective=objective,
                preparation=preparation,
                failure=final_execution.failure,
                last_coder_artifact=last_coder_artifact,
                last_synth_artifact=last_synth_artifact,
            )
            repair_contexts.append(context)
            self._emit(f"[REPAIR_START] attempt={attempt} category={context.failure.category.value}")
            state.current_phase = "repair"

            try:
                repair_response = self.repair_generator(context)
            except Exception as exc:
                state.status = "failed"
                state.stop_reason = "repair_generation_failed"
                state.current_phase = "failed"
                state.history.append(
                    RepairHistoryEntry(
                        attempt=attempt,
                        phase="repair_generation",
                        failure_category=context.failure.category.value,
                        failing_test=context.failing_test,
                        message=str(exc),
                    )
                )
                break
            self._emit(f"[REPAIR_COMPLETE] attempt={attempt}")
            last_coder_artifact = repair_response
            final_patch = self._apply_patch(
                response_text=repair_response,
                preparation=preparation,
                state=state,
                attempt=attempt,
            )
            if not final_patch.success:
                state.status = "failed"
                state.stop_reason = "patch_apply_failed"
                state.current_phase = "failed"
                break

            final_execution = self._run_tests(
                command=test_command,
                validation_commands=validation_commands,
                test_env=test_env,
                state=state,
                attempt=attempt,
            )
            if final_execution.passed:
                self._mark_success(state)
                break

        if not state.success and state.stop_reason is None:
            state.status = "failed"
            state.stop_reason = "repair_limit_reached"
            state.current_phase = "failed"

        return RepairConvergenceResult(
            state=state,
            initial_patch=initial_patch,
            final_patch=final_patch,
            final_execution=final_execution,
            repair_contexts=repair_contexts,
            telemetry=self._telemetry,
        )

    def _apply_patch(
        self,
        *,
        response_text: str,
        preparation: RepositoryExecutionPreparation,
        state: ConvergenceState,
        attempt: int,
    ) -> RepositoryExecutionApplyResult:
        state.current_phase = "patch_apply"
        try:
            result = self.repository_engine.apply_primary_output(
                response_text=response_text,
                plan=preparation.plan,
            )
        except (RepositoryExecutionError, ValueError) as exc:
            result = RepositoryExecutionApplyResult(
                summary=str(exc),
                results=[PatchResult(file_path="PRIMARY_CODER", success=False, error=str(exc))],
            )
        self._emit(f"[PATCH_APPLIED] attempt={attempt} success={result.success}")
        state.history.append(
            RepairHistoryEntry(
                attempt=attempt,
                phase="patch_apply",
                patch_success=result.success,
                message=result.summary,
            )
        )
        return result

    def _run_tests(
        self,
        *,
        command: list[str],
        validation_commands: list[list[str]] | None,
        test_env: dict[str, str] | None,
        state: ConvergenceState,
        attempt: int,
    ) -> TestExecutionResult:
        state.current_phase = "test_execution"
        state.iteration_count += 1
        commands = validation_commands or [command]
        self._emit(f"[TEST_START] attempt={attempt} command={' && '.join(' '.join(item) for item in commands)}")
        effective_env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            **(test_env or {}),
        }
        aggregate_stdout: list[str] = []
        aggregate_stderr: list[str] = []
        execution: ExecutionResult | None = None
        for item in commands:
            try:
                current = self.runner.run(command=item, cwd=str(self.repo_root), env=effective_env)
            except ExecutionRunnerError as exc:
                current = ExecutionResult(
                    command=item,
                    return_code=-127,
                    stdout="",
                    stderr=str(exc),
                    duration_seconds=0.0,
                    metadata={"runner_error": True},
                )
            aggregate_stdout.append(current.stdout)
            aggregate_stderr.append(current.stderr)
            execution = current
            if current.failed:
                break
        if execution is None:
            execution = ExecutionResult(command=command, return_code=0, stdout="", stderr="")
        if len(commands) > 1:
            execution = ExecutionResult(
                command=execution.command,
                return_code=execution.return_code,
                stdout="\n".join(part for part in aggregate_stdout if part),
                stderr="\n".join(part for part in aggregate_stderr if part),
                duration_seconds=execution.duration_seconds,
                metadata={"validation_commands": commands},
            )
        failure = self.classifier.classify(execution)
        result = TestExecutionResult.from_execution(execution, failure)
        state.test_pass_rate = _test_pass_rate(execution)
        if failure:
            state.last_failure_type = failure.category
            state.last_failing_test = failure.failing_tests[0] if failure.failing_tests else failure.file
            self._emit(f"[TEST_FAIL] attempt={attempt} category={failure.category.value}")
        else:
            self._emit(f"[TEST_PASS] attempt={attempt}")
        state.history.append(
            RepairHistoryEntry(
                attempt=attempt,
                phase="test_execution",
                failure_category=failure.category.value if failure else None,
                failing_test=state.last_failing_test,
                tests_passed=result.passed,
                test_pass_rate=state.test_pass_rate,
                message=f"return_code={result.return_code}",
            )
        )
        return result

    def _mark_success(self, state: ConvergenceState) -> None:
        state.success = True
        state.status = "converged"
        state.stop_reason = "tests_passed"
        state.current_phase = "converged"
        state.test_pass_rate = 1.0
        self._emit("[CONVERGED] tests_passed")

    def _emit(self, message: str) -> None:
        self._telemetry.append(message)
        if self._telemetry_sink:
            self._telemetry_sink(message)


def _test_pass_rate(result: ExecutionResult) -> float:
    if result.succeeded:
        return 1.0
    output = f"{result.stdout}\n{result.stderr}"
    passed = _summary_count(output, "passed")
    failed = (
        _summary_count(output, "failed")
        + _summary_count(output, "error")
        + _summary_count(output, "errors")
    )
    total = passed + failed
    if total <= 0:
        return 0.0
    return round(passed / total, 4)


def _summary_count(output: str, word: str) -> int:
    matches = re.findall(rf"(\d+)\s+{re.escape(word)}\b", output, flags=re.IGNORECASE)
    return sum(int(value) for value in matches)
