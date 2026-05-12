"""Repository intelligence orchestration service."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter

from backend.config.settings import Settings, get_settings
from backend.core.errors import ConfigurationError
from backend.core.logging import log_event
from backend.repointel.ast.parser import TreeSitterAstEngine
from backend.repointel.chunking import AstAwareChunker
from backend.repointel.context_builder import ContextBuilder
from backend.repointel.embeddings import EmbeddingService
from backend.repointel.graph import SymbolGraphEngine
from backend.repointel.models import ContextPackage, ExecutionPlan, IndexingStats
from backend.repointel.planner import PlanningLayer
from backend.repointel.retrieval import HybridRetrievalEngine
from backend.repointel.scanner import RepositoryScanner
from backend.repointel.vector_store import QdrantVectorStore
from backend.repointel.watcher import RepositoryWatcher

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexingDiagnostics:
    stats: IndexingStats
    scan_latency_ms: float
    parsing_latency_ms: float
    embedding_latency_ms: float
    vector_latency_ms: float


class RepositoryIntelligenceEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._scanner = RepositoryScanner(self._settings)
        self._ast = TreeSitterAstEngine(self._settings)
        self._graph = SymbolGraphEngine()
        self._chunker = AstAwareChunker()
        self._embeddings = EmbeddingService(self._settings)
        self._vector_store = QdrantVectorStore(self._settings)
        self._retrieval = HybridRetrievalEngine(self._settings, self._embeddings, self._vector_store)
        self._context_builder = ContextBuilder(self._settings, self._retrieval, self._graph)
        self._planner = PlanningLayer(self._context_builder)
        self._watcher: RepositoryWatcher | None = None
        self._index_lock = asyncio.Lock()

    @property
    def parser(self) -> TreeSitterAstEngine:
        return self._ast

    @property
    def embeddings(self) -> EmbeddingService:
        return self._embeddings

    @property
    def retrieval(self) -> HybridRetrievalEngine:
        return self._retrieval

    @property
    def context_builder(self) -> ContextBuilder:
        return self._context_builder

    @property
    def planner(self) -> PlanningLayer:
        return self._planner

    @property
    def vector_store(self) -> QdrantVectorStore:
        return self._vector_store

    async def index_repository(self, root: str | None = None) -> IndexingStats:
        diagnostics = await self.index_repository_with_diagnostics(root)
        return diagnostics.stats

    async def index_repository_with_diagnostics(self, root: str | None = None) -> IndexingDiagnostics:
        root = root or self._settings.repo_default_root
        if not root:
            raise ConfigurationError("Repository root must be provided for indexing.")

        async with self._index_lock:
            scan_started = perf_counter()
            scan_result = await self._scanner.scan(root)
            scan_latency_ms = (perf_counter() - scan_started) * 1000
            await self._vector_store.delete_paths(scan_result.deleted_paths)
            self._retrieval.remove_paths(scan_result.deleted_paths)
            for deleted_path in scan_result.deleted_paths:
                self._graph.remove_file(deleted_path)

            stats = IndexingStats(files_scanned=len(scan_result.manifest))
            all_chunks = []
            parsing_started = perf_counter()
            for repo_file in scan_result.files:
                self._graph.remove_file(repo_file.path)
                parsed = await asyncio.to_thread(self._ast.parse, repo_file)
                self._graph.upsert_parsed_file(parsed)
                chunks = self._chunker.chunk(parsed)
                all_chunks.extend(chunks)
                stats.files_indexed += 1
                stats.symbols_extracted += len(parsed.symbols)
                stats.chunks_indexed += len(chunks)
            parsing_latency_ms = (perf_counter() - parsing_started) * 1000

            embedding_latency_ms = 0.0
            vector_latency_ms = 0.0
            if all_chunks:
                embedding_started = perf_counter()
                vectors = await self._embeddings.embed_chunks(all_chunks)
                embedding_latency_ms = (perf_counter() - embedding_started) * 1000
                vector_started = perf_counter()
                await self._vector_store.upsert_chunks(all_chunks, vectors)
                vector_latency_ms = (perf_counter() - vector_started) * 1000
                self._retrieval.rebuild_lexical_index(all_chunks)

            self._scanner.save_manifest(scan_result.manifest)
            log_event(
                logger,
                logging.INFO,
                "repo.index.completed",
                "Repository indexing completed.",
                root=root,
                files_scanned=stats.files_scanned,
                files_indexed=stats.files_indexed,
                symbols_extracted=stats.symbols_extracted,
                chunks_indexed=stats.chunks_indexed,
                deleted_paths=len(scan_result.deleted_paths),
                scan_latency_ms=round(scan_latency_ms, 3),
                parsing_latency_ms=round(parsing_latency_ms, 3),
                embedding_latency_ms=round(embedding_latency_ms, 3),
                vector_latency_ms=round(vector_latency_ms, 3),
            )
            return IndexingDiagnostics(
                stats=stats,
                scan_latency_ms=round(scan_latency_ms, 3),
                parsing_latency_ms=round(parsing_latency_ms, 3),
                embedding_latency_ms=round(embedding_latency_ms, 3),
                vector_latency_ms=round(vector_latency_ms, 3),
            )

    async def build_context(self, query: str) -> ContextPackage:
        return await self._context_builder.build(query)

    async def plan_changes(self, query: str) -> ExecutionPlan:
        return await self._planner.plan(query)

    async def start_watcher(self, root: str | None = None) -> None:
        root = root or self._settings.repo_default_root
        if not root:
            raise ValueError("Repository root must be provided.")
        self._watcher = RepositoryWatcher(
            root=root,
            on_change=lambda: self.index_repository(root),
            debounce_ms=self._settings.repo_watcher_debounce_ms,
        )
        await self._watcher.start()

    async def stop_watcher(self) -> None:
        if self._watcher is not None:
            await self._watcher.stop()
            self._watcher = None

    async def shutdown(self) -> None:
        await self.stop_watcher()
        await self._vector_store.close()

    async def verify_runtime(self) -> dict[str, object]:
        language_status = self._ast.validate_languages()
        embedding_status = await self._embeddings.healthcheck()
        qdrant_status = await self._vector_store.healthcheck()
        retrieval_status = await self._retrieval.healthcheck()
        return {
            "tree_sitter_languages": {language.value: name for language, name in language_status.items()},
            "embedding": embedding_status,
            "qdrant": qdrant_status,
            "retrieval": retrieval_status,
        }
