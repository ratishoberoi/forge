from __future__ import annotations

import asyncio

from backend.config.settings import Settings
from backend.repointel.context_builder import ContextBuilder
from backend.repointel.graph import SymbolGraphEngine
from backend.repointel.models import CodeChunk, CodeSymbol, Language, RetrievalHit, SymbolKind
from backend.repointel.planner import PlanningLayer
from backend.repointel.retrieval import BM25Index, HybridRetrievalEngine


class StubEmbeddings:
    async def embed_texts(self, texts: dict[str, str]) -> dict[str, list[float]]:
        return {key: [1.0, 0.0, 0.0] for key in texts}


class StubVectorStore:
    async def query(self, vector: list[float], limit: int) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id="chunk-1",
                file_path="app.py",
                symbol_name="build_app",
                language=Language.PYTHON,
                score=0.9,
                vector_score=0.9,
                content="def build_app(): pass",
            )
        ]


def test_bm25_index_scores_relevant_chunk() -> None:
    index = BM25Index()
    index.rebuild(
        [
            CodeChunk(
                id="chunk-1",
                file_path="app.py",
                language=Language.PYTHON,
                content="def build_app(): pass",
                start_line=1,
                end_line=1,
            ),
            CodeChunk(
                id="chunk-2",
                file_path="db.py",
                language=Language.PYTHON,
                content="def connect_database(): pass",
                start_line=1,
                end_line=1,
            ),
        ]
    )
    hits = index.search("build app", limit=2)
    assert hits[0].chunk_id == "chunk-1"


def test_planning_layer_uses_context_hits() -> None:
    settings = Settings()
    retrieval = HybridRetrievalEngine(settings, StubEmbeddings(), StubVectorStore())  # type: ignore[arg-type]
    retrieval.rebuild_lexical_index(
        [
            CodeChunk(
                id="chunk-1",
                file_path="app.py",
                language=Language.PYTHON,
                symbol_id="app.py:build_app:1",
                symbol_name="build_app",
                content="def build_app(): pass",
                start_line=1,
                end_line=1,
            )
        ]
    )
    graph = SymbolGraphEngine()
    graph._symbol_index["app.py:build_app:1"] = CodeSymbol(  # type: ignore[attr-defined]
        id="app.py:build_app:1",
        name="build_app",
        kind=SymbolKind.FUNCTION,
        language=Language.PYTHON,
        file_path="app.py",
        start_line=1,
        end_line=1,
    )
    context_builder = ContextBuilder(settings, retrieval, graph)
    planner = PlanningLayer(context_builder)
    plan = asyncio.run(planner.plan("build app"))
    assert plan.impacted_files == ["app.py"]
    assert plan.steps
