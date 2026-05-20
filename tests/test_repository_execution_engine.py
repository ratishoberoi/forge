from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from backend.config.settings import Settings
from backend.runtime.execution_runner import ExecutionRunner
from backend.runtime.repository_execution_engine import RepositoryExecutionEngine


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        repo_index_state_path=str(tmp_path / ".forge" / "index_state.json"),
        repo_incremental=False,
    )


def test_repository_scan_detects_python_pytest_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='calculator'\n[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))

    prep = asyncio.run(engine.prepare("Build calculator app"))

    assert prep.scan.primary_language == "python"
    assert "pytest" in prep.scan.test_frameworks
    assert "pip/pyproject" in prep.scan.package_managers
    assert "app.py" in prep.scan.entrypoints
    assert prep.scan.build_commands


def test_context_builder_gathers_relevant_files_without_full_repo_dump(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "auth.py").write_text("def login():\n    return True\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    for index in range(12):
        (tmp_path / f"irrelevant_{index}.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))

    prep = asyncio.run(engine.prepare("Improve calculator subtraction"))

    assert "calculator.py" in prep.context.relevant_files
    assert "tests/test_calculator.py" in prep.context.related_tests
    assert len(prep.context.file_summaries) <= engine.MAX_CONTEXT_FILES
    assert "auth.py" not in prep.context.file_summaries


def test_planning_stage_for_build_calculator_creates_source_and_tests(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='calculator'\n", encoding="utf-8")
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))

    prep = asyncio.run(engine.prepare("Build calculator app"))

    assert "calculator.py" in prep.plan.files_to_create
    assert "tests/test_calculator.py" in prep.plan.files_to_create
    assert "tests/test_calculator.py" in prep.plan.expected_tests
    assert prep.plan.steps


def test_build_calculator_app_can_write_source_tests_and_pass(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='calculator'\n", encoding="utf-8")
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))
    prep = asyncio.run(engine.prepare("Build calculator app"))
    response = json.dumps(
        {
            "summary": "Build calculator module and tests",
            "files": {
                "calculator.py": (
                    "def add(a: int, b: int) -> int:\n"
                    "    return a + b\n\n"
                    "def subtract(a: int, b: int) -> int:\n"
                    "    return a - b\n"
                ),
                "tests/test_calculator.py": (
                    "from calculator import add, subtract\n\n"
                    "def test_add():\n"
                    "    assert add(2, 3) == 5\n\n"
                    "def test_subtract():\n"
                    "    assert subtract(5, 3) == 2\n"
                ),
            },
        }
    )

    result = engine.apply_primary_output(response_text=response, plan=prep.plan)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    tests = ExecutionRunner(timeout=30).run(
        command=["pytest", "-q"],
        cwd=str(tmp_path),
        env=env,
    )

    assert result.success is True
    assert (tmp_path / "calculator.py").exists()
    assert (tmp_path / "tests" / "test_calculator.py").exists()
    assert tests.succeeded, tests.stderr
