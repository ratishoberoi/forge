from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.runtime.git_manager import GitManager, GitManagerError
from backend.runtime.repository_rag import RepositoryRAG
from backend.runtime.safe_tools import SafeToolExecutor, SafeToolError
from backend.runtime.semantic_memory import SemanticMemory
from backend.runtime.validation_suite import BuildValidator


@dataclass(slots=True)
class ToolActivity:
    tool: str
    action: str
    status: str
    detail: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "action": self.action,
            "status": self.status,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


class RepositoryTool:
    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()

    def inspect(self) -> dict[str, Any]:
        files = [
            path.relative_to(self.repo_root).as_posix()
            for path in self.repo_root.rglob("*")
            if path.is_file()
            and not any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in path.relative_to(self.repo_root).parts)
        ]
        return {"root": str(self.repo_root), "files": len(files), "sample": files[:50]}


class GitTool:
    def __init__(self, repo_root: str) -> None:
        self.repo_root = repo_root

    def status(self) -> dict[str, Any]:
        try:
            return GitManager(self.repo_root).status().to_dict()
        except GitManagerError as exc:
            return {"error": str(exc)}


class BuildTool:
    def __init__(self, repo_root: str) -> None:
        self.validator = BuildValidator(repo_root=repo_root)

    def validate(self) -> dict[str, Any]:
        return self.validator.validate().to_dict()


class TestTool(BuildTool):
    pass


class SearchTool:
    def __init__(self, repo_root: str, memory: SemanticMemory | None = None) -> None:
        self.rag = RepositoryRAG(repo_root=repo_root, memory=memory)

    def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return [hit.to_dict() for hit in self.rag.retrieve(query=query, limit=limit)]


class MemoryTool:
    def __init__(self, repo_root: str, memory: SemanticMemory | None = None) -> None:
        self.repo_root = repo_root
        self.memory = memory or SemanticMemory()

    def retrieve(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return [
            item.to_dict()
            for item in self.memory.retrieve(repository_path=self.repo_root, query=query, limit=limit)
        ]


class DocumentationTool(SearchTool):
    def search_docs(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return [
            hit
            for hit in self.search(query, limit=limit)
            if hit.get("metadata", {}).get("role") == "documentation"
        ]


class ToolManager:
    """Unified observable local tool ecosystem."""

    def __init__(self, *, repo_root: str, memory: SemanticMemory | None = None) -> None:
        self.repo_root = str(Path(repo_root).resolve())
        self.memory = memory or SemanticMemory()
        self.activities: list[ToolActivity] = []
        self.repository = RepositoryTool(self.repo_root)
        self.git = GitTool(self.repo_root)
        self.build = BuildTool(self.repo_root)
        self.test = TestTool(self.repo_root)
        self.search = SearchTool(self.repo_root, self.memory)
        self.memory_tool = MemoryTool(self.repo_root, self.memory)
        self.documentation = DocumentationTool(self.repo_root, self.memory)
        self.executor = SafeToolExecutor(repo_root=self.repo_root)

    def inspect_repository(self) -> dict[str, Any]:
        return self._record("RepositoryTool", "inspect", lambda: self.repository.inspect())

    def git_status(self) -> dict[str, Any]:
        return self._record("GitTool", "status", lambda: self.git.status())

    def search_repository(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return self._record("SearchTool", "search", lambda: self.search.search(query, limit=limit))

    def retrieve_memory(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return self._record("MemoryTool", "retrieve", lambda: self.memory_tool.retrieve(query, limit=limit))

    def run_command(self, command: list[str]) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            result = self.executor.run(command)
            return result.to_dict()

        return self._record("BuildTool", "run_command", call)

    def snapshot(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "tools": [
                "RepositoryTool",
                "GitTool",
                "BuildTool",
                "TestTool",
                "SearchTool",
                "MemoryTool",
                "DocumentationTool",
            ],
            "activities": [activity.to_dict() for activity in self.activities[-50:]],
        }

    def _record(self, tool: str, action: str, callback):
        try:
            result = callback()
            self.activities.append(ToolActivity(tool=tool, action=action, status="ok"))
            return result
        except SafeToolError as exc:
            self.activities.append(ToolActivity(tool=tool, action=action, status="blocked", detail=str(exc)))
            raise
        except Exception as exc:
            self.activities.append(ToolActivity(tool=tool, action=action, status="failed", detail=str(exc)))
            raise
