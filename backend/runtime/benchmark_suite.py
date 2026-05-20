from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from backend.runtime.repository_bootstrap import RepositoryBootstrap
from backend.runtime.validation_suite import AcceptanceValidator, BuildValidator, QualityScorer, VisualValidator


@dataclass(slots=True)
class BenchmarkCase:
    name: str
    objective: str
    expected_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objective": self.objective,
            "expected_files": self.expected_files,
        }


@dataclass(slots=True)
class BenchmarkResult:
    case: BenchmarkCase
    workspace: str
    success: bool
    repair_rate: float
    convergence_rate: float
    completion_rate: float
    quality_score: dict[str, Any]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "workspace": self.workspace,
            "success": self.success,
            "repair_rate": self.repair_rate,
            "convergence_rate": self.convergence_rate,
            "completion_rate": self.completion_rate,
            "quality_score": self.quality_score,
            "details": self.details,
        }


class BenchmarkSuite:
    """Runs disposable benchmarks under an isolated benchmark root."""

    DEFAULT_CASES = [
        BenchmarkCase("build-calculator", "Build calculator", ["README.md"]),
        BenchmarkCase("build-rest-api", "Build REST API", ["README.md"]),
        BenchmarkCase("add-authentication", "Add authentication", ["README.md"]),
        BenchmarkCase("create-dashboard-page", "Create dashboard page", ["README.md"]),
        BenchmarkCase("add-database-model", "Add database model", ["README.md"]),
    ]

    def __init__(self, *, root: str = ".forge/benchmarks") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        cases: list[BenchmarkCase] | None = None,
        cleanup: bool = True,
        runner: Callable[[Path, BenchmarkCase], dict[str, Any]] | None = None,
    ) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for case in cases or self.DEFAULT_CASES:
            workspace = self.root / case.name
            if workspace.exists():
                shutil.rmtree(workspace)
            workspace.mkdir(parents=True)
            RepositoryBootstrap(str(workspace)).bootstrap_if_needed(case.objective)
            run_result = runner(workspace, case) if runner else {"tests_passed": True, "repair_attempts": 0, "converged": True}
            build = BuildValidator(repo_root=str(workspace)).validate()
            acceptance = AcceptanceValidator().validate(
                repo_root=str(workspace),
                objective=case.objective,
                tests_passed=build.passed and bool(run_result.get("tests_passed", True)),
                changed_files=[path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()],
                expected_files=case.expected_files,
            )
            visual = VisualValidator().validate(repo_root=str(workspace))
            quality = QualityScorer().score(
                repo_root=str(workspace),
                acceptance=acceptance,
                build=build,
                visual=visual,
                repair_attempts=int(run_result.get("repair_attempts", 0)),
                changed_files=acceptance.details.get("changed_files", []),
            )
            success = acceptance.passed and build.passed
            results.append(
                BenchmarkResult(
                    case=case,
                    workspace=str(workspace),
                    success=success,
                    repair_rate=1.0 if int(run_result.get("repair_attempts", 0)) > 0 else 0.0,
                    convergence_rate=1.0 if run_result.get("converged", success) else 0.0,
                    completion_rate=1.0 if success else 0.0,
                    quality_score=quality.to_dict(),
                    details={
                        "acceptance": acceptance.to_dict(),
                        "build": build.to_dict(),
                        "visual": visual.to_dict(),
                    },
                )
            )
            if cleanup:
                shutil.rmtree(workspace, ignore_errors=True)
        return results
