from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from backend.config.settings import Settings
from backend.runtime.autonomous_repair import (
    AutonomousRepairConvergenceEngine,
    FailureClassifier,
    FailureType,
)
from backend.runtime.execution_result import ExecutionResult
from backend.runtime.repository_execution_engine import RepositoryExecutionEngine


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        repo_index_state_path=str(tmp_path / ".forge" / "index_state.json"),
        repo_incremental=False,
    )


def test_failure_classifier_extracts_pytest_failure() -> None:
    result = ExecutionResult(
        command=["pytest", "-q"],
        return_code=1,
        stdout=(
            "FAILED tests/test_calculator.py::test_add - AssertionError: assert -1 == 3\n"
            "1 failed, 1 passed in 0.03s\n"
        ),
        stderr="",
    )

    failure = FailureClassifier().classify(result)

    assert failure is not None
    assert failure.category == FailureType.ASSERTION_ERROR
    assert failure.file == "tests/test_calculator.py"
    assert failure.failing_tests == ["tests/test_calculator.py::test_add"]


def test_broken_calculator_is_repaired_until_tests_pass(tmp_path: Path) -> None:
    _write_python_project(tmp_path)
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))
    preparation = asyncio.run(engine.prepare("Build calculator app"))
    telemetry: list[str] = []

    repair_engine = AutonomousRepairConvergenceEngine(
        repo_root=str(tmp_path),
        repository_engine=engine,
        telemetry=telemetry.append,
        repair_generator=lambda _: _calculator_patch(add_expression="a + b"),
    )

    result = repair_engine.run(
        objective="Build calculator app",
        preparation=preparation,
        initial_response_text=_calculator_patch(add_expression="a - b"),
        test_command=[sys.executable, "-m", "pytest", "-q"],
        max_repairs=1,
    )

    assert result.converged is True
    assert result.state.stop_reason == "tests_passed"
    assert result.state.repair_count == 1
    assert result.final_execution is not None
    assert result.final_execution.passed is True
    assert (tmp_path / "calculator.py").read_text(encoding="utf-8").count("return a + b") == 1
    assert "[TEST_FAIL] attempt=0 category=AssertionError" in result.telemetry
    assert "[REPAIR_START] attempt=1 category=AssertionError" in result.telemetry
    assert "[CONVERGED] tests_passed" in result.telemetry
    assert telemetry == result.telemetry


def test_irreparable_calculator_stops_at_repair_limit(tmp_path: Path) -> None:
    _write_python_project(tmp_path)
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))
    preparation = asyncio.run(engine.prepare("Build calculator app"))

    repair_engine = AutonomousRepairConvergenceEngine(
        repo_root=str(tmp_path),
        repository_engine=engine,
        repair_generator=lambda _: _calculator_patch(add_expression="a - b"),
    )

    result = repair_engine.run(
        objective="Build calculator app",
        preparation=preparation,
        initial_response_text=_calculator_patch(add_expression="a - b"),
        test_command=[sys.executable, "-m", "pytest", "-q"],
        max_repairs=2,
    )

    assert result.converged is False
    assert result.state.status == "failed"
    assert result.state.stop_reason == "repair_limit_reached"
    assert result.state.repair_count == 2
    assert result.state.last_failure_type == FailureType.ASSERTION_ERROR
    assert result.final_execution is not None
    assert result.final_execution.failed is True
    assert result.repair_contexts
    assert result.repair_contexts[-1].failing_test == "tests/test_calculator.py::test_add"


def _write_python_project(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "[project]\nname='calculator'\n[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )


def _calculator_patch(*, add_expression: str) -> str:
    return json.dumps(
        {
            "summary": "Calculator implementation",
            "files": {
                "calculator.py": (
                    "def add(a: int, b: int) -> int:\n"
                    f"    return {add_expression}\n\n"
                    "def subtract(a: int, b: int) -> int:\n"
                    "    return a - b\n"
                ),
                "tests/test_calculator.py": (
                    "from calculator import add, subtract\n\n"
                    "def test_add():\n"
                    "    assert add(1, 2) == 3\n\n"
                    "def test_subtract():\n"
                    "    assert subtract(5, 3) == 2\n"
                ),
            },
        }
    )
