from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from backend.runtime.execution_result import ExecutionResult
from backend.runtime.framework_detection import FrameworkDetector
from backend.runtime.safe_tools import SafeToolExecutor, SafeToolError


@dataclass(slots=True)
class ValidationResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "errors": self.errors,
        }


class AcceptanceValidator:
    """Checks whether generated repository state satisfies objective-level criteria."""

    def validate(
        self,
        *,
        repo_root: str,
        objective: str,
        tests_passed: bool,
        changed_files: list[str],
        expected_files: list[str] | None = None,
    ) -> ValidationResult:
        root = Path(repo_root).resolve()
        errors: list[str] = []
        expected = expected_files or self._expected_files(objective)
        missing = [path for path in expected if not (root / path).exists()]
        if missing:
            errors.append(f"missing expected files: {missing}")
        if not tests_passed:
            errors.append("tests did not pass")
        required_terms = self._required_terms(objective)
        if required_terms and not self._repository_contains(root, required_terms):
            errors.append(f"objective terms not found in generated files: {required_terms}")
        route_errors = self._route_errors(root, objective)
        errors.extend(route_errors)
        return ValidationResult(
            name="acceptance",
            passed=not errors,
            details={
                "expected_files": expected,
                "changed_files": changed_files,
                "required_terms": required_terms,
            },
            errors=errors,
        )

    @staticmethod
    def _expected_files(objective: str) -> list[str]:
        lowered = objective.lower()
        if any(term in lowered for term in ("website", "landing", "dashboard", "crm", "saas")):
            return ["README.md"]
        if any(term in lowered for term in ("api", "oauth", "auth", "stripe")):
            return ["README.md"]
        return []

    @staticmethod
    def _required_terms(objective: str) -> list[str]:
        terms = []
        for term in ("oauth", "stripe", "dashboard", "auth", "login", "crm", "tenant"):
            if term in objective.lower():
                terms.append(term)
        return terms

    @staticmethod
    def _repository_contains(root: Path, terms: list[str]) -> bool:
        searchable = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".md"}:
                searchable.append(path.read_text(encoding="utf-8", errors="replace").lower())
        combined = "\n".join(searchable)
        return all(term in combined for term in terms)

    @staticmethod
    def _route_errors(root: Path, objective: str) -> list[str]:
        lowered = objective.lower()
        if not any(term in lowered for term in ("api", "oauth", "auth", "login")):
            return []
        source = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in root.rglob("*.py")
            if path.is_file()
        )
        if any(pattern in source for pattern in ("@app.get", "@app.post", "Blueprint(", "urlpatterns")):
            return []
        return ["required API or route definitions were not found"]


class BuildValidator:
    """Runs safe build and test commands and reports structured failures."""

    def __init__(self, *, repo_root: str, executor: SafeToolExecutor | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.executor = executor or SafeToolExecutor(repo_root=str(self.repo_root))
        self.detector = FrameworkDetector()

    def validate(self, commands: list[str] | None = None) -> ValidationResult:
        profile = self.detector.detect(self.repo_root)
        command_texts = commands or profile.test_commands + profile.build_commands
        if not command_texts:
            command_texts = ["python -m compileall ."] if profile.language == "python" else []
        results: list[ExecutionResult] = []
        errors: list[str] = []
        for command_text in command_texts:
            command = command_text.split()
            try:
                result = self.executor.run(command, cwd=str(self.repo_root))
            except SafeToolError as exc:
                errors.append(str(exc))
                continue
            results.append(result)
            if result.failed:
                errors.append(f"{command_text} failed with exit code {result.return_code}")
        return ValidationResult(
            name="build",
            passed=not errors,
            details={
                "commands": command_texts,
                "results": [result.to_dict() for result in results],
            },
            errors=errors,
        )


class VisualValidator:
    """Lightweight web validation for pages, routes, and renderable components."""

    def validate(self, *, repo_root: str, urls: list[str] | None = None) -> ValidationResult:
        root = Path(repo_root).resolve()
        errors: list[str] = []
        details: dict[str, Any] = {"files": [], "urls": []}
        web_files = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".html", ".jsx", ".tsx", ".js", ".ts"}
        ]
        details["files"] = web_files
        if web_files and not any(path.endswith(("index.html", "page.tsx", "App.jsx", "App.tsx")) for path in web_files):
            errors.append("no primary web page/component found")
        for url in urls or []:
            try:
                with urlopen(url, timeout=5) as response:
                    details["urls"].append({"url": url, "status": response.status})
                    if response.status >= 400:
                        errors.append(f"{url} returned {response.status}")
            except Exception as exc:
                errors.append(f"{url} failed to load: {exc}")
        return ValidationResult(name="visual", passed=not errors, details=details, errors=errors)


@dataclass(slots=True)
class QualityScore:
    architecture: float
    tests: float
    maintainability: float
    complexity: float
    documentation: float
    convergence: float

    @property
    def overall(self) -> float:
        return round(
            (
                self.architecture
                + self.tests
                + self.maintainability
                + self.complexity
                + self.documentation
                + self.convergence
            )
            / 6,
            2,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "architecture": self.architecture,
            "tests": self.tests,
            "maintainability": self.maintainability,
            "complexity": self.complexity,
            "documentation": self.documentation,
            "convergence": self.convergence,
            "overall": self.overall,
        }


class QualityScorer:
    def score(
        self,
        *,
        repo_root: str,
        acceptance: ValidationResult,
        build: ValidationResult,
        visual: ValidationResult,
        repair_attempts: int,
        changed_files: list[str],
    ) -> QualityScore:
        root = Path(repo_root).resolve()
        has_readme = (root / "README.md").exists()
        tests = 10.0 if build.passed else 3.0
        architecture = 8.0 if acceptance.passed else 5.0
        maintainability = max(3.0, 10.0 - min(len(changed_files), 20) * 0.2)
        complexity = max(3.0, 10.0 - max(0, len(changed_files) - 10) * 0.3)
        documentation = 9.0 if has_readme else 4.0
        convergence = max(2.0, 10.0 - repair_attempts * 1.5)
        if not visual.passed:
            architecture = min(architecture, 6.0)
        return QualityScore(
            architecture=round(architecture, 2),
            tests=round(tests, 2),
            maintainability=round(maintainability, 2),
            complexity=round(complexity, 2),
            documentation=round(documentation, 2),
            convergence=round(convergence, 2),
        )


def split_commands(command_texts: list[str]) -> list[list[str]]:
    return [re.split(r"\s+", command.strip()) for command in command_texts if command.strip()]
