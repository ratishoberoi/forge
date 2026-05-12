"""Subsystem verification helpers for repository intelligence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from backend.config.settings import Settings, get_settings
from backend.repointel.models import CodeChunk, Language, RepositoryFile
from backend.repointel.service import RepositoryIntelligenceEngine


@dataclass(slots=True)
class DiagnosticCheck:
    name: str
    status: str
    detail: str


class RepositoryIntelligenceDiagnostics:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._engine = RepositoryIntelligenceEngine(self._settings)

    async def run(self, repo_root: str | None = None) -> list[DiagnosticCheck]:
        checks: list[DiagnosticCheck] = []
        sample_root = repo_root
        temp_dir: TemporaryDirectory[str] | None = None
        if sample_root is None:
            temp_dir = TemporaryDirectory()
            sample_root = self._build_sample_repo(Path(temp_dir.name))

        try:
            checks.append(await self._run_check("scanner", self._verify_scanner, sample_root))
            checks.append(await self._run_check("tree_sitter", self._verify_ast))
            checks.append(await self._run_check("embeddings", self._verify_embeddings))
            checks.append(await self._run_check("qdrant", self._verify_qdrant))
            checks.append(await self._run_check("indexing", self._verify_indexing, sample_root))
            checks.append(await self._run_check("retrieval", self._verify_retrieval, sample_root))
            checks.append(await self._run_check("context_builder", self._verify_context, sample_root))
            checks.append(await self._run_check("planning", self._verify_planning, sample_root))
        finally:
            await self._engine.shutdown()
            if temp_dir is not None:
                temp_dir.cleanup()
        return checks

    async def _run_check(self, name: str, fn, *args: object) -> DiagnosticCheck:
        started = perf_counter()
        try:
            detail = await fn(*args)
            latency_ms = round((perf_counter() - started) * 1000, 3)
            return DiagnosticCheck(name=name, status="ok", detail=f"{detail} ({latency_ms} ms)")
        except Exception as exc:
            latency_ms = round((perf_counter() - started) * 1000, 3)
            return DiagnosticCheck(name=name, status="failed", detail=f"{exc} ({latency_ms} ms)")

    async def _verify_scanner(self, repo_root: str) -> str:
        scan_result = await self._engine._scanner.scan(repo_root)
        return f"scanned {len(scan_result.manifest)} files"

    async def _verify_ast(self) -> str:
        validated = self._engine._ast.validate_languages()
        return f"validated {len(validated)} languages"

    async def _verify_embeddings(self) -> str:
        status = await self._engine._embeddings.healthcheck()
        return f"embedding dimension {status['embedding_dimension']}"

    async def _verify_qdrant(self) -> str:
        status = await self._engine._vector_store.healthcheck()
        return f"collections={status['collections']}"

    async def _verify_indexing(self, repo_root: str) -> str:
        diagnostics = await self._engine.index_repository_with_diagnostics(repo_root)
        return (
            f"indexed {diagnostics.stats.files_indexed} files, "
            f"{diagnostics.stats.chunks_indexed} chunks"
        )

    async def _verify_retrieval(self, repo_root: str) -> str:
        await self._engine.index_repository(repo_root)
        hits = await self._engine.build_context("where is auth middleware initialized?")
        return f"retrieved {len(hits.hits)} hits"

    async def _verify_context(self, repo_root: str) -> str:
        await self._engine.index_repository(repo_root)
        context = await self._engine.build_context("where is auth middleware initialized?")
        return f"context files={len(context.related_files)} symbols={len(context.related_symbols)}"

    async def _verify_planning(self, repo_root: str) -> str:
        await self._engine.index_repository(repo_root)
        plan = await self._engine.plan_changes("add caching to auth middleware")
        return f"plan impacted_files={len(plan.impacted_files)} steps={len(plan.steps)}"

    def _build_sample_repo(self, root: Path) -> str:
        repo_root = root / "repo"
        repo_root.mkdir(parents=True, exist_ok=True)
        (repo_root / "app.py").write_text(
            "from util import auth_middleware\n\n"
            "def build_app():\n"
            "    return auth_middleware()\n",
            encoding="utf-8",
        )
        (repo_root / "util.py").write_text(
            "def auth_middleware():\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )
        return str(repo_root)
