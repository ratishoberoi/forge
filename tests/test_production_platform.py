from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.runtime.benchmark_suite import BenchmarkCase, BenchmarkSuite
from backend.runtime.release_report import ReleaseReportStore, build_release_report
from backend.runtime.repository_bootstrap import RepositoryBootstrap
from backend.runtime.safe_tools import SafeToolError, SafeToolExecutor
from backend.runtime.validation_suite import (
    AcceptanceValidator,
    BuildValidator,
    QualityScorer,
    VisualValidator,
)


def test_empty_repository_bootstrap_generates_web_project(tmp_path: Path) -> None:
    result = RepositoryBootstrap(str(tmp_path)).bootstrap_if_needed("Build a SaaS landing page")

    assert result.applied is True
    assert "React" in result.framework.frameworks
    assert (tmp_path / "package.json").exists()
    assert (tmp_path / "src" / "App.jsx").exists()
    assert (tmp_path / "README.md").exists()


def test_empty_repository_bootstrap_generates_fastapi_project(tmp_path: Path) -> None:
    result = RepositoryBootstrap(str(tmp_path)).bootstrap_if_needed("Build REST API")

    assert result.applied is True
    assert "FastAPI" in result.framework.frameworks
    assert (tmp_path / "app" / "main.py").exists()
    assert (tmp_path / "tests" / "test_app.py").exists()
    assert (tmp_path / "Dockerfile").exists()


def test_build_acceptance_visual_and_quality_validation(tmp_path: Path) -> None:
    RepositoryBootstrap(str(tmp_path)).bootstrap_if_needed("Build a website")

    build = BuildValidator(repo_root=str(tmp_path)).validate()
    acceptance = AcceptanceValidator().validate(
        repo_root=str(tmp_path),
        objective="Build a website",
        tests_passed=build.passed,
        changed_files=["src/App.jsx", "README.md"],
        expected_files=["README.md", "src/App.jsx"],
    )
    visual = VisualValidator().validate(repo_root=str(tmp_path))
    quality = QualityScorer().score(
        repo_root=str(tmp_path),
        acceptance=acceptance,
        build=build,
        visual=visual,
        repair_attempts=0,
        changed_files=["src/App.jsx", "README.md"],
    )

    assert build.passed is True
    assert acceptance.passed is True
    assert visual.passed is True
    assert quality.overall >= 8.0


def test_acceptance_rejects_fastapi_todo_placeholder(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='todo'\ndependencies=['fastapi','uvicorn','pytest']\n",
        encoding="utf-8",
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
        "@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n\n"
        "def objective_summary():\n    return 'Build a complete FastAPI Todo application'\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_app.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Todo placeholder\n", encoding="utf-8")

    acceptance = AcceptanceValidator().validate(
        repo_root=str(tmp_path),
        objective="Build a complete FastAPI Todo application",
        tests_passed=True,
        changed_files=["app/main.py", "README.md", "tests/test_app.py"],
    )

    assert acceptance.passed is False
    assert any("missing expected files" in error for error in acceptance.errors)
    assert any("routes" in error or "models" in error for error in acceptance.errors)


def test_safe_tool_executor_blocks_workspace_escape(tmp_path: Path) -> None:
    executor = SafeToolExecutor(repo_root=str(tmp_path))

    with pytest.raises(SafeToolError):
        executor.run(["python", "-c", "print('bad')"], cwd="/tmp")


def test_release_report_persists_run_summary(tmp_path: Path) -> None:
    report = build_release_report(
        run_id="run-1",
        objective="Build website",
        result={
            "task_plan": {"tasks": []},
            "changed_files": ["README.md"],
            "tests_passed": True,
            "return_code": 0,
            "final_verdict": "approved",
            "quality_score": {"overall": 9.0},
        },
    )
    store = ReleaseReportStore(str(tmp_path / "reports"))
    path = store.write(report)

    assert path.exists()
    assert store.read("run-1")["quality_score"]["overall"] == 9.0


def test_benchmark_suite_runs_in_isolated_workspace_and_cleans_up(tmp_path: Path) -> None:
    root = tmp_path / "benchmarks"
    results = BenchmarkSuite(root=str(root)).run(
        cases=[BenchmarkCase("rest-api", "Build REST API", ["README.md"])],
        cleanup=True,
    )

    assert len(results) == 1
    assert results[0].success is True
    assert not (root / "rest-api").exists()
    assert all("backend/" not in result.workspace and "frontend/" not in result.workspace for result in results)
