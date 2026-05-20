from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.config.settings import Settings, get_settings
from backend.repointel.models import Language
from backend.repointel.scanner import RepositoryScanner
from backend.runtime.architecture_memory import (
    ArchitectureMemory,
    ArchitectureMemoryRecord,
    build_dependency_graph,
    dependency_closure,
)
from backend.runtime.context_compressor import CompressedRepositoryContext, ContextCompressor
from backend.runtime.long_horizon import LongHorizonExecutionGraph, LongHorizonPreparation
from backend.runtime.output_parser import OutputParser
from backend.runtime.patch_writer import PatchResult, PatchWriter
from backend.runtime.repo_workspace import RepositoryWorkspace
from backend.runtime.task_planner import TaskPlan, TaskPlanner


class RepositoryExecutionError(Exception):
    """Raised when repository execution preparation or writes fail."""


@dataclass(slots=True)
class RepositoryIntelligenceSummary:
    root: str
    languages: list[str]
    primary_language: str
    frameworks: list[str]
    package_managers: list[str]
    test_frameworks: list[str]
    entrypoints: list[str]
    build_commands: list[str]
    test_commands: list[str]
    architecture_summary: str
    files_scanned: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "languages": self.languages,
            "primary_language": self.primary_language,
            "frameworks": self.frameworks,
            "package_managers": self.package_managers,
            "test_frameworks": self.test_frameworks,
            "entrypoints": self.entrypoints,
            "build_commands": self.build_commands,
            "test_commands": self.test_commands,
            "architecture_summary": self.architecture_summary,
            "files_scanned": self.files_scanned,
        }


@dataclass(slots=True)
class TargetedRepositoryContext:
    objective: str
    relevant_files: list[str]
    imports: dict[str, list[str]]
    dependencies: list[str]
    related_tests: list[str]
    file_summaries: dict[str, str]
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    compressed_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "relevant_files": self.relevant_files,
            "imports": self.imports,
            "dependencies": self.dependencies,
            "related_tests": self.related_tests,
            "file_summaries": self.file_summaries,
            "dependency_graph": self.dependency_graph,
            "compressed_summary": self.compressed_summary,
        }


@dataclass(slots=True)
class RepositoryExecutionPlan:
    objective: str
    files_to_modify: list[str]
    files_to_create: list[str]
    risks: list[str]
    expected_tests: list[str]
    steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "files_to_modify": self.files_to_modify,
            "files_to_create": self.files_to_create,
            "risks": self.risks,
            "expected_tests": self.expected_tests,
            "steps": self.steps,
        }


@dataclass(slots=True)
class RepositoryExecutionPreparation:
    scan: RepositoryIntelligenceSummary
    context: TargetedRepositoryContext
    plan: RepositoryExecutionPlan
    task_plan: TaskPlan | None = None
    execution_graph: LongHorizonExecutionGraph | None = None
    architecture_memory: ArchitectureMemoryRecord | None = None
    compressed_context: CompressedRepositoryContext | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan": self.scan.to_dict(),
            "context": self.context.to_dict(),
            "plan": self.plan.to_dict(),
            "task_plan": self.task_plan.to_dict() if self.task_plan else None,
            "execution_graph": self.execution_graph.to_dict() if self.execution_graph else None,
            "architecture_memory": self.architecture_memory.to_dict() if self.architecture_memory else None,
            "compressed_context": self.compressed_context.to_dict() if self.compressed_context else None,
        }

    def to_prompt_context(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass(slots=True)
class RepositoryExecutionApplyResult:
    summary: str
    results: list[PatchResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.results) and all(result.success for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "success": self.success,
            "results": [
                {
                    "file_path": result.file_path,
                    "success": result.success,
                    "resolved_path": str(result.resolved_path) if result.resolved_path else None,
                    "error": result.error,
                }
                for result in self.results
            ],
        }


class RepositoryExecutionEngine:
    """
    Repository-aware execution preparation for autonomous coding.
    It runs before courtroom cognition and builds bounded context, not a full repo dump.
    """

    MAX_CONTEXT_FILES = 8
    MAX_FILE_SUMMARY_CHARS = 1600

    def __init__(
        self,
        *,
        repo_root: str,
        settings: Settings | None = None,
        writer: PatchWriter | None = None,
        architecture_memory: ArchitectureMemory | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.exists() or not self.repo_root.is_dir():
            raise RepositoryExecutionError(f"Repository root does not exist: {self.repo_root}")
        self.settings = settings or get_settings()
        self.scanner = RepositoryScanner(self.settings)
        self.workspace = RepositoryWorkspace(str(self.repo_root))
        self.writer = writer or PatchWriter(self.workspace, backup=True)
        self.parser = OutputParser()
        self.architecture_memory = architecture_memory or ArchitectureMemory()
        self.task_planner = TaskPlanner()
        self.context_compressor = ContextCompressor()

    async def prepare(self, objective: str) -> RepositoryExecutionPreparation:
        if not objective.strip():
            raise RepositoryExecutionError("objective must not be blank.")
        scan_result = await self.scanner.scan(str(self.repo_root))
        files = scan_result.files
        summary = self._build_summary(files)
        context = self._build_context(objective, files, summary)
        plan = self._build_plan(objective, summary, context)
        memory = self.architecture_memory.upsert_from_preparation(
            repository_path=str(self.repo_root),
            scan=summary,
            context=context,
        )
        task_plan = self.task_planner.plan(
            objective=objective,
            preparation=RepositoryExecutionPreparation(scan=summary, context=context, plan=plan),
            memory=memory,
        )
        execution_graph = LongHorizonExecutionGraph.from_task_plan(task_plan)
        compressed_context = self.context_compressor.compress(
            objective=objective,
            architecture_summary=summary.architecture_summary,
            file_summaries=context.file_summaries,
            dependency_graph=memory.dependency_graph,
            related_tests=context.related_tests,
            memory=memory,
        )
        context.compressed_summary = compressed_context.to_dict()
        return RepositoryExecutionPreparation(
            scan=summary,
            context=context,
            plan=plan,
            task_plan=task_plan,
            execution_graph=execution_graph,
            architecture_memory=memory,
            compressed_context=compressed_context,
        )

    def apply_primary_output(
        self,
        *,
        response_text: str,
        plan: RepositoryExecutionPlan,
    ) -> RepositoryExecutionApplyResult:
        parsed = self.parser.parse_primary_output(response_text)
        allowed_paths = set(plan.files_to_create) | set(plan.files_to_modify)
        if not allowed_paths:
            allowed_paths = set(parsed.files)

        unexpected = sorted(set(parsed.files) - allowed_paths)
        if unexpected:
            raise RepositoryExecutionError(
                f"Generated files are outside the approved execution plan: {unexpected}"
            )

        results = [
            self.writer.apply(file_path=path, new_content=content)
            for path, content in parsed.files.items()
        ]
        return RepositoryExecutionApplyResult(summary=parsed.summary, results=results)

    def _build_summary(self, files) -> RepositoryIntelligenceSummary:
        language_counts = Counter(file.language.value for file in files)
        languages = sorted(language_counts)
        root_files = {path.name for path in self.repo_root.iterdir() if path.is_file()}
        primary_language = language_counts.most_common(1)[0][0] if language_counts else "unknown"
        if primary_language == "unknown" and (
            "pyproject.toml" in root_files or "requirements.txt" in root_files
        ):
            primary_language = "python"
            languages = ["python"]
        all_paths = [file.path for file in files]
        frameworks = self._detect_frameworks(root_files, all_paths)
        package_managers = self._detect_package_managers(root_files)
        test_frameworks = self._detect_test_frameworks(root_files, all_paths, primary_language)
        entrypoints = self._detect_entrypoints(root_files, all_paths)
        build_commands = self._detect_build_commands(root_files, frameworks, package_managers)
        test_commands = self._detect_test_commands(root_files, test_frameworks, package_managers)
        return RepositoryIntelligenceSummary(
            root=str(self.repo_root),
            languages=languages,
            primary_language=primary_language,
            frameworks=frameworks,
            package_managers=package_managers,
            test_frameworks=test_frameworks,
            entrypoints=entrypoints,
            build_commands=build_commands,
            test_commands=test_commands,
            architecture_summary=self._architecture_summary(
                primary_language,
                frameworks,
                all_paths,
            ),
            files_scanned=len(files),
        )

    def _build_context(
        self,
        objective: str,
        files,
        summary: RepositoryIntelligenceSummary,
    ) -> TargetedRepositoryContext:
        query_terms = set(_tokens(objective))
        scored: list[tuple[int, str, Any]] = []
        for file in files:
            path_terms = set(_tokens(file.path))
            content_terms = set(_tokens(file.content[:4000]))
            score = len(query_terms & path_terms) * 4 + len(query_terms & content_terms)
            if _is_test_path(file.path):
                score += 1
            if _is_entrypoint(file.path):
                score += 2
            scored.append((score, file.path, file))

        selected = [
            item[2]
            for item in sorted(scored, key=lambda item: (-item[0], item[1]))
            if item[0] > 0
        ][: self.MAX_CONTEXT_FILES]
        if not selected:
            selected = sorted(files, key=lambda file: file.path)[: self.MAX_CONTEXT_FILES]

        related_tests = sorted(
            file.path for file in files if _is_test_path(file.path)
        )[: self.MAX_CONTEXT_FILES]
        imports = {
            file.path: _extract_imports(file.content, file.language)
            for file in selected
        }
        dependency_graph = build_dependency_graph(
            {
                file.path: _extract_imports(file.content, file.language)
                for file in files
            },
            [file.path for file in files],
        )
        expanded_paths = dependency_closure(
            dependency_graph,
            [file.path for file in selected],
            max_depth=2,
            max_files=self.MAX_CONTEXT_FILES * 3,
        )
        by_path = {file.path: file for file in files}
        expanded = [by_path[path] for path in expanded_paths if path in by_path]
        if expanded:
            selected = expanded[: self.MAX_CONTEXT_FILES * 3]
            imports = {
                file.path: _extract_imports(file.content, file.language)
                for file in selected
            }
        selected_paths = [file.path for file in selected]
        dependency_targets = [
            target
            for path in selected_paths
            for target in dependency_graph.get(path, [])
            if target in by_path and target not in selected_paths
        ]
        if dependency_targets:
            selected.extend(by_path[path] for path in dependency_targets[: self.MAX_CONTEXT_FILES])
            imports = {
                file.path: _extract_imports(file.content, file.language)
                for file in selected
            }
        dependencies = sorted(
            set(summary.frameworks + summary.package_managers + summary.test_frameworks)
        )
        file_summaries = {
            file.path: file.content[: self.MAX_FILE_SUMMARY_CHARS]
            for file in selected
        }
        return TargetedRepositoryContext(
            objective=objective,
            relevant_files=[file.path for file in selected],
            imports=imports,
            dependencies=dependencies,
            related_tests=related_tests,
            file_summaries=file_summaries,
            dependency_graph={
                path: dependency_graph.get(path, [])
                for path in [file.path for file in selected]
            },
        )

    def _build_plan(
        self,
        objective: str,
        summary: RepositoryIntelligenceSummary,
        context: TargetedRepositoryContext,
    ) -> RepositoryExecutionPlan:
        objective_lower = objective.lower()
        files_to_modify = list(context.relevant_files)
        files_to_create: list[str] = []
        expected_tests = list(context.related_tests)

        if "calculator" in objective_lower:
            if summary.primary_language == "python":
                files_to_create = _append_missing(files_to_create, ["calculator.py", "tests/test_calculator.py"])
                expected_tests = _append_missing(expected_tests, ["tests/test_calculator.py"])
            elif summary.primary_language in {"javascript", "typescript"} or "next.js" in summary.frameworks:
                files_to_create = _append_missing(files_to_create, ["app/page.tsx", "app/calculator.test.tsx"])
                expected_tests = _append_missing(expected_tests, ["app/calculator.test.tsx"])

        if not files_to_create and not files_to_modify:
            default_file = "app.py" if summary.primary_language in {"python", "unknown"} else "src/index.ts"
            files_to_create.append(default_file)

        risks = []
        if not summary.test_frameworks:
            risks.append("No test framework detected; generated tests may require dependency setup.")
        if not summary.entrypoints:
            risks.append("No clear application entrypoint detected.")

        steps = [
            "Use repository scan summary to respect detected language, framework, and package manager.",
            "Edit only files in files_to_modify or files_to_create.",
            "Create or update tests listed in expected_tests.",
            "Return PRIMARY_CODER JSON with complete file contents for every generated file.",
        ]
        return RepositoryExecutionPlan(
            objective=objective,
            files_to_modify=files_to_modify,
            files_to_create=files_to_create,
            risks=risks,
            expected_tests=expected_tests,
            steps=steps,
        )

    @staticmethod
    def _detect_frameworks(root_files: set[str], paths: list[str]) -> list[str]:
        frameworks: list[str] = []
        package_json = "package.json" in root_files
        pyproject = "pyproject.toml" in root_files
        if "next.config.js" in root_files or "next.config.mjs" in root_files:
            frameworks.append("Next.js")
        if any(path.startswith("app/") for path in paths) and package_json:
            frameworks.append("React")
        if any(path.endswith("manage.py") for path in paths):
            frameworks.append("Django")
        if any("fastapi" in path.lower() for path in paths) or pyproject:
            frameworks.append("Python")
        return sorted(dict.fromkeys(frameworks))

    @staticmethod
    def _detect_package_managers(root_files: set[str]) -> list[str]:
        managers = []
        if "package-lock.json" in root_files:
            managers.append("npm")
        if "pnpm-lock.yaml" in root_files:
            managers.append("pnpm")
        if "yarn.lock" in root_files:
            managers.append("yarn")
        if "pyproject.toml" in root_files:
            managers.append("pip/pyproject")
        if "requirements.txt" in root_files:
            managers.append("pip")
        if "Cargo.toml" in root_files:
            managers.append("cargo")
        if "go.mod" in root_files:
            managers.append("go")
        return managers

    @staticmethod
    def _detect_test_frameworks(
        root_files: set[str],
        paths: list[str],
        primary_language: str,
    ) -> list[str]:
        frameworks = []
        if primary_language == "python" and (
            "pytest.ini" in root_files
            or "pyproject.toml" in root_files
            or any(path.startswith("tests/") and path.endswith(".py") for path in paths)
        ):
            frameworks.append("pytest")
        if "package.json" in root_files:
            if any(path.endswith((".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx")) for path in paths):
                frameworks.append("vitest/jest")
        return frameworks

    @staticmethod
    def _detect_entrypoints(root_files: set[str], paths: list[str]) -> list[str]:
        candidates = [
            "main.py",
            "app.py",
            "run.py",
            "backend/app.py",
            "src/index.ts",
            "src/main.ts",
            "app/page.tsx",
        ]
        return [candidate for candidate in candidates if candidate in root_files or candidate in paths]

    @staticmethod
    def _detect_build_commands(
        root_files: set[str],
        frameworks: list[str],
        package_managers: list[str],
    ) -> list[str]:
        commands = []
        if "npm" in package_managers:
            commands.append("npm run build")
        if "cargo" in package_managers:
            commands.append("cargo build")
        if "go" in package_managers:
            commands.append("go build ./...")
        if "pip/pyproject" in package_managers and "Python" in frameworks:
            commands.append("python -m compileall .")
        return commands

    @staticmethod
    def _detect_test_commands(
        root_files: set[str],
        test_frameworks: list[str],
        package_managers: list[str],
    ) -> list[str]:
        commands = []
        if "pytest" in test_frameworks:
            commands.append("pytest -q")
        if "vitest/jest" in test_frameworks:
            commands.append("npm test")
        if "cargo" in package_managers:
            commands.append("cargo test")
        if "go" in package_managers:
            commands.append("go test ./...")
        return commands

    @staticmethod
    def _architecture_summary(
        primary_language: str,
        frameworks: list[str],
        paths: list[str],
    ) -> str:
        top_dirs = sorted({path.split("/", 1)[0] for path in paths if "/" in path})[:8]
        framework_text = ", ".join(frameworks) if frameworks else "no dominant framework"
        dirs_text = ", ".join(top_dirs) if top_dirs else "flat repository"
        return (
            f"Primary language is {primary_language}; detected {framework_text}; "
            f"top-level structure: {dirs_text}."
        )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _is_test_path(path: str) -> bool:
    lower = path.lower()
    return lower.startswith("tests/") or "_test." in lower or ".test." in lower or ".spec." in lower


def _is_entrypoint(path: str) -> bool:
    return path in {"main.py", "app.py", "run.py", "backend/app.py", "src/index.ts", "src/main.ts", "app/page.tsx"}


def _extract_imports(content: str, language: Language) -> list[str]:
    imports: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if language is Language.PYTHON and (stripped.startswith("import ") or stripped.startswith("from ")):
            imports.append(stripped)
        elif language in {Language.JAVASCRIPT, Language.TYPESCRIPT} and stripped.startswith("import "):
            imports.append(stripped)
    return imports[:20]


def _append_missing(items: list[str], additions: list[str]) -> list[str]:
    result = list(items)
    for item in additions:
        if item not in result:
            result.append(item)
    return result
