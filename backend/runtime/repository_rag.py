from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.runtime.semantic_memory import SemanticMemory, SemanticMemoryItem


@dataclass(slots=True)
class RepositoryRAGHit:
    path: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RepositoryRAGIndexResult:
    repository_path: str
    indexed_files: int
    skipped_files: int
    total_files: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_path": self.repository_path,
            "indexed_files": self.indexed_files,
            "skipped_files": self.skipped_files,
            "total_files": self.total_files,
        }


class RepositoryRAG:
    """Local repository retrieval for source, tests, configs, and documentation."""

    SUPPORTED_SUFFIXES = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".css",
        ".html",
        ".md",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".sql",
    }
    SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".next", "dist", "build"}

    def __init__(self, *, repo_root: str, memory: SemanticMemory | None = None, max_file_chars: int = 6000) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.memory = memory or SemanticMemory()
        self.max_file_chars = max_file_chars

    def index(self, *, max_files: int = 2000) -> RepositoryRAGIndexResult:
        self.memory.delete_repository_kind(repository_path=str(self.repo_root), kind="repository_file")
        indexed = 0
        skipped = 0
        total = 0
        for path in self._iter_files():
            total += 1
            if indexed >= max_files:
                skipped += 1
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped += 1
                continue
            relative = path.relative_to(self.repo_root).as_posix()
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            self.memory.upsert(
                repository_path=str(self.repo_root),
                kind="repository_file",
                text=f"{relative}\n{text[: self.max_file_chars]}",
                metadata={
                    "repository_path": str(self.repo_root),
                    "path": relative,
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "role": _file_role(relative),
                },
                item_id=f"repo:{self.repo_root}:{relative}:{digest}",
            )
            indexed += 1
        return RepositoryRAGIndexResult(
            repository_path=str(self.repo_root),
            indexed_files=indexed,
            skipped_files=skipped,
            total_files=total,
        )

    def index_documents(self, documents: dict[str, str], *, max_files: int = 2000) -> RepositoryRAGIndexResult:
        self.memory.delete_repository_kind(repository_path=str(self.repo_root), kind="repository_file")
        indexed = 0
        skipped = 0
        for relative, text in sorted(documents.items()):
            if indexed >= max_files:
                skipped += 1
                continue
            if Path(relative).suffix not in self.SUPPORTED_SUFFIXES:
                skipped += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            self.memory.upsert(
                repository_path=str(self.repo_root),
                kind="repository_file",
                text=f"{relative}\n{text[: self.max_file_chars]}",
                metadata={
                    "repository_path": str(self.repo_root),
                    "path": relative,
                    "sha256": digest,
                    "size_bytes": len(text.encode("utf-8")),
                    "role": _file_role(relative),
                },
                item_id=f"repo:{self.repo_root}:{relative}:{digest}",
            )
            indexed += 1
        return RepositoryRAGIndexResult(
            repository_path=str(self.repo_root),
            indexed_files=indexed,
            skipped_files=skipped,
            total_files=len(documents),
        )

    def retrieve(self, *, query: str, limit: int = 10) -> list[RepositoryRAGHit]:
        items = self.memory.retrieve(
            repository_path=str(self.repo_root),
            query=query,
            kinds=["repository_file"],
            limit=limit,
        )
        return [hit for item in items if (hit := self._hit_from_item(item)).path]

    def stats(self) -> dict[str, Any]:
        stats = self.memory.stats(repository_path=str(self.repo_root))
        stats["repository_path"] = str(self.repo_root)
        return stats

    def _iter_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.repo_root.rglob("*"):
            if any(part in self.SKIP_DIRS for part in path.relative_to(self.repo_root).parts):
                continue
            if path.is_file() and path.suffix in self.SUPPORTED_SUFFIXES:
                files.append(path)
        return sorted(files)

    def _hit_from_item(self, item: SemanticMemoryItem) -> RepositoryRAGHit:
        expected_repo = str(self.repo_root)
        if item.repository_path != expected_repo:
            return RepositoryRAGHit(path="", content="", score=0.0, metadata={"isolation_error": True})
        path = str(item.metadata.get("path", ""))
        content = item.text.split("\n", 1)[1] if "\n" in item.text else item.text
        return RepositoryRAGHit(path=path, content=content, score=item.score, metadata=item.metadata)


def _file_role(path: str) -> str:
    lowered = path.lower()
    if lowered.startswith("tests/") or ".test." in lowered or ".spec." in lowered:
        return "test"
    if lowered.endswith((".md", ".txt")):
        return "documentation"
    if lowered.endswith((".json", ".toml", ".yaml", ".yml")):
        return "configuration"
    if lowered.endswith(".sql"):
        return "database"
    return "source"
