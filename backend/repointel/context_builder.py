"""Context assembly using retrieval and graph expansion."""

from __future__ import annotations

from backend.config.settings import Settings, get_settings
from backend.repointel.graph import SymbolGraphEngine
from backend.repointel.models import ContextPackage
from backend.repointel.retrieval import HybridRetrievalEngine


class ContextBuilder:
    def __init__(
        self,
        settings: Settings | None = None,
        retrieval: HybridRetrievalEngine | None = None,
        graph: SymbolGraphEngine | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._retrieval = retrieval or HybridRetrievalEngine(self._settings)
        self._graph = graph or SymbolGraphEngine()

    async def build(self, query: str) -> ContextPackage:
        hits = await self._retrieval.retrieve(query, limit=self._settings.repo_retrieval_limit)
        related_files = list(dict.fromkeys(hit.file_path for hit in hits))
        related_symbols = self._graph.symbols_for_paths(related_files)
        dependency_neighbors = {
            file_path: self._graph.neighbors(file_path, limit=self._settings.repo_graph_neighbors)
            for file_path in related_files
        }
        return ContextPackage(
            query=query,
            hits=hits,
            related_symbols=related_symbols,
            related_files=related_files,
            dependency_neighbors=dependency_neighbors,
        )
