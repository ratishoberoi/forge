from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
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
from backend.runtime.adr import ADRStore
from backend.runtime.context_assembly import AssembledContext, ContextAssemblyEngine
from backend.runtime.context_compressor import CompressedRepositoryContext, ContextCompressor
from backend.runtime.knowledge_graph import KnowledgeGraphRecord, KnowledgeGraphStore
from backend.runtime.long_horizon import LongHorizonExecutionGraph, LongHorizonPreparation
from backend.runtime.output_parser import OutputParser
from backend.runtime.patch_writer import PatchResult, PatchWriter
from backend.runtime.project_brain import ProjectBrain, ProjectBrainRecord
from backend.runtime.repository_rag import RepositoryRAG, RepositoryRAGIndexResult
from backend.runtime.repo_workspace import RepositoryWorkspace
from backend.runtime.semantic_memory import SemanticMemory, SemanticMemoryItem
from backend.runtime.task_planner import TaskPlan, TaskPlanner
from backend.runtime.tool_manager import ToolManager


class RepositoryExecutionError(Exception):
    """Raised when repository execution preparation or writes fail."""


class ObjectiveType(StrEnum):
    PATCH = "PATCH"
    FEATURE = "FEATURE"
    APPLICATION = "APPLICATION"
    REFACTOR = "REFACTOR"
    MIGRATION = "MIGRATION"


@dataclass(slots=True)
class AcceptanceContract:
    objective_type: ObjectiveType
    required_files: list[str] = field(default_factory=list)
    required_tests: list[str] = field(default_factory=list)
    required_routes: list[str] = field(default_factory=list)
    required_models: list[str] = field(default_factory=list)
    required_dependencies: list[str] = field(default_factory=list)
    required_terms: list[str] = field(default_factory=list)
    minimum_source_files: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_type": self.objective_type.value,
            "required_files": self.required_files,
            "required_tests": self.required_tests,
            "required_routes": self.required_routes,
            "required_models": self.required_models,
            "required_dependencies": self.required_dependencies,
            "required_terms": self.required_terms,
            "minimum_source_files": self.minimum_source_files,
        }


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
    objective_type: ObjectiveType = ObjectiveType.PATCH
    acceptance_contract: AcceptanceContract | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "objective_type": self.objective_type.value,
            "files_to_modify": self.files_to_modify,
            "files_to_create": self.files_to_create,
            "risks": self.risks,
            "expected_tests": self.expected_tests,
            "steps": self.steps,
            "acceptance_contract": (
                self.acceptance_contract.to_dict()
                if self.acceptance_contract
                else None
            ),
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
    project_brain: ProjectBrainRecord | None = None
    semantic_memories: list[SemanticMemoryItem] = field(default_factory=list)
    repository_rag: dict[str, Any] | None = None
    context_assembly: AssembledContext | None = None
    knowledge_graph: KnowledgeGraphRecord | None = None
    adrs: list[dict[str, Any]] = field(default_factory=list)
    tool_activity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan": self.scan.to_dict(),
            "context": self.context.to_dict(),
            "plan": self.plan.to_dict(),
            "task_plan": self.task_plan.to_dict() if self.task_plan else None,
            "execution_graph": self.execution_graph.to_dict() if self.execution_graph else None,
            "architecture_memory": self.architecture_memory.to_dict() if self.architecture_memory else None,
            "compressed_context": self.compressed_context.to_dict() if self.compressed_context else None,
            "project_brain": self.project_brain.to_dict() if self.project_brain else None,
            "semantic_memories": [item.to_dict() for item in self.semantic_memories],
            "repository_rag": self.repository_rag,
            "context_assembly": self.context_assembly.to_dict() if self.context_assembly else None,
            "knowledge_graph": self.knowledge_graph.to_dict() if self.knowledge_graph else None,
            "adrs": self.adrs,
            "tool_activity": self.tool_activity,
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
        self.project_brain = ProjectBrain()
        self.semantic_memory = SemanticMemory()
        self.adr_store = ADRStore()
        self.knowledge_graph_store = KnowledgeGraphStore()
        self.context_assembly = ContextAssemblyEngine()

    async def prepare(
        self,
        objective: str,
        telemetry_callback: Callable[[str], None] | None = None,
    ) -> RepositoryExecutionPreparation:
        if not objective.strip():
            raise RepositoryExecutionError("objective must not be blank.")
        telemetry = telemetry_callback or (lambda _message: None)
        objective_type = classify_objective(objective)
        telemetry(f"[ACTIVE_REPOSITORY] {self.repo_root}")
        telemetry(f"[OBJECTIVE_CLASSIFICATION] {objective_type.value}")
        scan_result = await self.scanner.scan(str(self.repo_root))
        files = scan_result.files
        summary = self._build_summary(files)
        context = self._build_context(objective, files, summary)
        plan = self._build_plan(objective, summary, context)
        telemetry(f"[REPOSITORY_SCAN] files={summary.files_scanned} language={summary.primary_language}")
        memory = self.architecture_memory.upsert_from_preparation(
            repository_path=str(self.repo_root),
            scan=summary,
            context=context,
        )
        telemetry(f"[MEMORY_STORE] architecture_memory repository={self.repo_root}")
        project_brain = self.project_brain.update_from_preparation(
            repository_path=str(self.repo_root),
            objective=objective,
            architecture_summary=summary.architecture_summary,
            service_boundaries=memory.service_boundaries,
            feature_summaries=[f"{path}: {context.file_summaries[path][:240]}" for path in context.relevant_files if path in context.file_summaries],
        )
        telemetry(f"[MEMORY_STORE] project_brain repository={self.repo_root}")
        self.semantic_memory.upsert(
            repository_path=str(self.repo_root),
            kind="objective",
            text=objective,
            metadata={"source": "repository_preparation"},
        )
        self.semantic_memory.upsert(
            repository_path=str(self.repo_root),
            kind="architecture",
            text=summary.architecture_summary,
            metadata={"frameworks": summary.frameworks, "language": summary.primary_language},
        )
        repository_rag = RepositoryRAG(repo_root=str(self.repo_root), memory=self.semantic_memory)
        rag_index = repository_rag.index_documents(
            {file.path: file.content for file in files},
            max_files=500,
        )
        telemetry(f"[RAG_STORE] repository={self.repo_root} indexed={rag_index.indexed_files}")
        repository_hits = repository_rag.retrieve(query=objective, limit=12)
        self._assert_repository_hits_isolated(repository_hits)
        semantic_hits = self.semantic_memory.retrieve(
            repository_path=str(self.repo_root),
            query=objective,
            kinds=["objective", "architecture", "failure", "repair", "implementation", "decision"],
            limit=10,
        )
        self._assert_semantic_hits_isolated(semantic_hits)
        adr_records = self.adr_store.infer_from_frameworks(
            repository_path=str(self.repo_root),
            frameworks=summary.frameworks,
            databases=[],
        )
        knowledge_graph = self.knowledge_graph_store.build_from_preparation(
            repository_path=str(self.repo_root),
            objective=objective,
            relevant_files=context.relevant_files,
            related_tests=context.related_tests,
            dependency_graph=memory.dependency_graph,
            service_boundaries=memory.service_boundaries,
        )
        telemetry(
            f"[GRAPH_STORE] repository={self.repo_root} "
            f"nodes={len(knowledge_graph.nodes)} edges={len(knowledge_graph.edges)}"
        )
        knowledge_nodes = self.knowledge_graph_store.relevant_nodes(
            repository_path=str(self.repo_root),
            query=objective,
            limit=20,
        )
        assembled = self.context_assembly.assemble(
            objective=objective,
            relevant_files=context.relevant_files,
            dependency_graph=memory.dependency_graph,
            architecture_memory=memory,
            project_brain=project_brain,
            semantic_memories=semantic_hits,
            repository_hits=repository_hits,
            adrs=adr_records,
            knowledge_graph=knowledge_graph,
            knowledge_nodes=knowledge_nodes,
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
        tool_manager = ToolManager(repo_root=str(self.repo_root), memory=self.semantic_memory)
        tool_manager.inspect_repository()
        tool_manager.search_repository(objective, limit=3)
        return RepositoryExecutionPreparation(
            scan=summary,
            context=context,
            plan=plan,
            task_plan=task_plan,
            execution_graph=execution_graph,
            architecture_memory=memory,
            compressed_context=compressed_context,
            project_brain=project_brain,
            semantic_memories=semantic_hits,
            repository_rag={
                "index": rag_index.to_dict(),
                "hits": [hit.to_dict() for hit in repository_hits],
                "stats": repository_rag.stats(),
            },
            context_assembly=assembled,
            knowledge_graph=knowledge_graph,
            adrs=[record.to_dict() for record in adr_records],
            tool_activity=tool_manager.snapshot(),
        )

    def _assert_repository_hits_isolated(self, hits: list[RepositoryRAGHit]) -> None:
        root = str(self.repo_root)
        for hit in hits:
            repository_path = str(hit.metadata.get("repository_path", root))
            if repository_path != root:
                raise RepositoryExecutionError(
                    f"RepositoryRAG returned cross-repository hit: {repository_path} != {root}"
                )

    def _assert_semantic_hits_isolated(self, hits: list[SemanticMemoryItem]) -> None:
        root = str(self.repo_root)
        for hit in hits:
            if hit.repository_path != root:
                raise RepositoryExecutionError(
                    f"SemanticMemory returned cross-repository hit: {hit.repository_path} != {root}"
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
        self._validate_generated_output(parsed.files, plan)

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
        if any("from fastapi import" in file.content or "import fastapi" in file.content for file in files):
            frameworks = _append_missing(frameworks, ["FastAPI"])
        if any("from flask import" in file.content or "import flask" in file.content for file in files):
            frameworks = _append_missing(frameworks, ["Flask"])
        if any("django" in file.content.lower() for file in files):
            frameworks = _append_missing(frameworks, ["Django"])
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
        objective_type = classify_objective(objective)
        acceptance_contract = build_acceptance_contract(objective, summary)
        files_to_modify = list(context.relevant_files)
        files_to_create: list[str] = []
        expected_tests = list(context.related_tests)

        if objective_type == ObjectiveType.APPLICATION and acceptance_contract:
            existing_required = [
                path
                for path in acceptance_contract.required_files
                if (self.repo_root / path).exists()
                and not self._is_bootstrap_placeholder(path)
            ]
            existing_dependency_files = [
                path
                for path in ("pyproject.toml", "requirements.txt", "package.json")
                if (self.repo_root / path).exists()
            ]
            missing_required = [
                path
                for path in acceptance_contract.required_files
                if not (self.repo_root / path).exists()
                or self._is_bootstrap_placeholder(path)
            ]
            placeholder_context_files = [
                path
                for path in files_to_modify
                if self._is_bootstrap_placeholder(path)
                and path not in missing_required
            ]
            files_to_modify = _dedupe(
                existing_required
                + existing_dependency_files
                + [
                    path
                    for path in files_to_modify
                    if not _is_bootstrap_placeholder_path(path)
                ]
                + placeholder_context_files
            )
            files_to_create = _append_missing(files_to_create, missing_required)
            expected_tests = _append_missing(expected_tests, acceptance_contract.required_tests)

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
        if acceptance_contract and acceptance_contract.required_routes:
            steps.append(
                "Implement acceptance contract routes: "
                + ", ".join(acceptance_contract.required_routes)
            )
        plan = RepositoryExecutionPlan(
            objective=objective,
            files_to_modify=files_to_modify,
            files_to_create=files_to_create,
            risks=risks,
            expected_tests=expected_tests,
            steps=steps,
            objective_type=objective_type,
            acceptance_contract=acceptance_contract,
        )
        self._validate_plan(plan)
        return plan

    def _validate_plan(self, plan: RepositoryExecutionPlan) -> None:
        if plan.objective_type != ObjectiveType.APPLICATION:
            return
        allowed = set(plan.files_to_modify) | set(plan.files_to_create)
        contract = plan.acceptance_contract
        required = set(contract.required_files if contract else [])
        missing_from_plan = sorted(required - allowed)
        if missing_from_plan:
            raise RepositoryExecutionError(
                "APPLICATION plan is missing required files: "
                + ", ".join(missing_from_plan)
            )
        if not plan.expected_tests:
            raise RepositoryExecutionError("APPLICATION plan must include tests.")
        source_files = [path for path in allowed if _is_source_path(path) and not _is_test_path(path)]
        if len(source_files) < (contract.minimum_source_files if contract else 1):
            raise RepositoryExecutionError("APPLICATION plan must include functional source files.")
        if _is_template_only_path_set(allowed):
            raise RepositoryExecutionError(
                "APPLICATION plan only targets bootstrap/template files."
            )

    def _validate_generated_output(
        self,
        files: dict[str, str],
        plan: RepositoryExecutionPlan,
    ) -> None:
        if plan.objective_type != ObjectiveType.APPLICATION:
            return
        changed = set(files)
        if _is_template_only_path_set(changed):
            raise RepositoryExecutionError(
                "APPLICATION output only changes bootstrap/template files."
            )
        if not any(_is_source_path(path) and not _is_test_path(path) for path in changed):
            raise RepositoryExecutionError("APPLICATION output contains no functional source files.")
        combined = self._combined_repository_text(files)
        contract = plan.acceptance_contract
        if contract:
            missing_files = [
                path
                for path in contract.required_files
                if path not in files and not (self.repo_root / path).exists()
            ]
            if missing_files:
                raise RepositoryExecutionError(
                    "APPLICATION output is missing required files: "
                    + ", ".join(missing_files)
                )
            missing_routes = [
                route for route in contract.required_routes if route not in combined
            ]
            if missing_routes:
                raise RepositoryExecutionError(
                    "APPLICATION output is missing required routes: "
                    + ", ".join(missing_routes)
                )
            missing_models = [
                model for model in contract.required_models if model.lower() not in combined.lower()
            ]
            if missing_models:
                raise RepositoryExecutionError(
                    "APPLICATION output is missing required models: "
                    + ", ".join(missing_models)
                )
        if _looks_like_placeholder(combined):
            raise RepositoryExecutionError("APPLICATION output appears to be a placeholder.")

    def _is_bootstrap_placeholder(self, relative_path: str) -> bool:
        path = self.repo_root / relative_path
        if not path.exists() or not path.is_file():
            return False
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return _is_bootstrap_placeholder_path(relative_path) and _looks_like_placeholder(content)

    def _combined_repository_text(self, files: dict[str, str]) -> str:
        parts = list(files.values())
        for path in (self.repo_root.rglob("*") if self.repo_root.exists() else []):
            if path.is_file() and path.suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".toml", ".json"}:
                relative = path.relative_to(self.repo_root).as_posix()
                if relative not in files:
                    parts.append(path.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(parts)

    @staticmethod
    def _detect_frameworks(root_files: set[str], paths: list[str]) -> list[str]:
        frameworks: list[str] = []
        package_json = "package.json" in root_files
        pyproject = "pyproject.toml" in root_files
        if "next.config.js" in root_files or "next.config.mjs" in root_files:
            frameworks.append("Next.js")
        if package_json and any(
            path.startswith("app/")
            or path.startswith("src/")
            or path.endswith((".tsx", ".jsx"))
            for path in paths
        ):
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
            commands.append("python -m pytest -q")
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


def classify_objective(objective: str) -> ObjectiveType:
    lowered = objective.lower()
    if any(term in lowered for term in ("migrate", "migration", "convert")):
        return ObjectiveType.MIGRATION
    if any(term in lowered for term in ("refactor", "restructure", "reorganize")):
        return ObjectiveType.REFACTOR
    if any(term in lowered for term in ("fix ", "bug", "repair", "failing", "error", "exception")):
        return ObjectiveType.PATCH
    application_nouns = {
        "app",
        "application",
        "website",
        "landing page",
        "dashboard",
        "crm",
        "saas",
        "todo",
        "calculator",
        "api",
        "service",
    }
    creation_verbs = {"build", "create", "generate", "scaffold", "develop"}
    if any(verb in lowered for verb in creation_verbs) and any(noun in lowered for noun in application_nouns):
        return ObjectiveType.APPLICATION
    if lowered.startswith(("add ", "implement ", "introduce ")):
        return ObjectiveType.FEATURE
    return ObjectiveType.PATCH


def build_acceptance_contract(
    objective: str,
    summary: RepositoryIntelligenceSummary | None = None,
) -> AcceptanceContract:
    objective_type = classify_objective(objective)
    lowered = objective.lower()
    if objective_type != ObjectiveType.APPLICATION:
        return AcceptanceContract(objective_type=objective_type)

    if "todo" in lowered and ("fastapi" in lowered or "api" in lowered or _has_framework(summary, "FastAPI")):
        return AcceptanceContract(
            objective_type=objective_type,
            required_files=[
                "requirements.txt",
                "app/__init__.py",
                "app/main.py",
                "app/models.py",
                "app/database.py",
                "app/schemas.py",
                "app/repository.py",
                "tests/test_todos.py",
                "README.md",
                "Dockerfile",
            ],
            required_tests=["tests/test_todos.py"],
            required_routes=["/todos", "/todos/{todo_id}"],
            required_models=["Todo"],
            required_dependencies=["fastapi", "uvicorn", "pytest"],
            required_terms=["todo", "create", "read", "update", "delete"],
            minimum_source_files=5,
        )

    if "calculator" in lowered:
        return AcceptanceContract(
            objective_type=objective_type,
            required_files=["calculator.py", "tests/test_calculator.py"],
            required_tests=["tests/test_calculator.py"],
            required_terms=["add", "subtract"],
            minimum_source_files=1,
        )

    if any(term in lowered for term in ("react", "landing", "website", "dashboard", "crm", "saas")):
        return AcceptanceContract(
            objective_type=objective_type,
            required_files=[
                "package.json",
                "index.html",
                "src/main.jsx",
                "src/App.jsx",
                "tests/app.test.mjs",
                "README.md",
            ],
            required_tests=["tests/app.test.mjs"],
            required_terms=["landing"] if "landing" in lowered else [],
            minimum_source_files=2,
        )

    if "api" in lowered or _has_framework(summary, "FastAPI"):
        return AcceptanceContract(
            objective_type=objective_type,
            required_files=[
                "pyproject.toml",
                "app/__init__.py",
                "app/main.py",
                "tests/test_app.py",
                "README.md",
                "Dockerfile",
            ],
            required_tests=["tests/test_app.py"],
            required_routes=["/health"],
            required_dependencies=["fastapi", "uvicorn", "pytest"],
            minimum_source_files=1,
        )

    return AcceptanceContract(
        objective_type=objective_type,
        required_files=["README.md"],
        required_tests=[],
        minimum_source_files=1,
    )


def _has_framework(summary: RepositoryIntelligenceSummary | None, framework: str) -> bool:
    return bool(summary and framework in summary.frameworks)


def _append_missing(items: list[str], additions: list[str]) -> list[str]:
    result = list(items)
    for item in additions:
        if item not in result:
            result.append(item)
    return result


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _is_source_path(path: str) -> bool:
    return Path(path).suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".html"}


def _is_template_only_path_set(paths: set[str]) -> bool:
    if not paths:
        return True
    template_paths = {
        "README.md",
        "Dockerfile",
        "app.py",
        "app/main.py",
        "tests/test_app.py",
    }
    return paths <= template_paths


def _is_bootstrap_placeholder_path(path: str) -> bool:
    return path in {
        "README.md",
        "Dockerfile",
        "app.py",
        "app/main.py",
        "tests/test_app.py",
    }


def _looks_like_placeholder(text: str) -> bool:
    lowered = text.lower()
    if "objective_summary" in lowered and not any(term in lowered for term in ("class todo", "/todos", "def add", "landing")):
        return True
    placeholder_terms = ("todo: implement", "not implemented", "placeholder", "coming soon")
    return any(term in lowered for term in placeholder_terms)
