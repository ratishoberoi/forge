from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.config.settings import Settings
from backend.repointel.context_builder import ContextBuilder
from backend.repointel.embeddings import EmbeddingService
from backend.repointel.graph import SymbolGraphEngine
from backend.repointel.planner import PlanningLayer
from backend.repointel.retrieval import HybridRetrievalEngine
from backend.repointel.scanner import RepositoryScanner
from backend.repointel.ast.parser import TreeSitterAstEngine
from backend.repointel.chunking import AstAwareChunker
from backend.repointel.vector_store import QdrantVectorStore


class FakeEmbeddingService(EmbeddingService):
    async def initialize(self) -> None:
        return None

    async def healthcheck(self) -> dict[str, object]:
        return {"model_name": "fake", "embedding_dimension": 3, "device": "cpu"}

    async def embed_text(self, text: str, *, cache_key: str | None = None) -> list[float]:
        return [1.0, 0.0, 0.0] if "auth" in text.lower() else [0.0, 1.0, 0.0]

    async def embed_texts(self, texts: dict[str, str]) -> dict[str, list[float]]:
        return {
            key: ([1.0, 0.0, 0.0] if "auth" in value.lower() else [0.0, 1.0, 0.0])
            for key, value in texts.items()
        }


def test_repository_intelligence_integration() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        root.mkdir()
        (root / "app.py").write_text(
            "from util import auth_middleware\n\n"
            "def build_app():\n"
            "    return auth_middleware()\n",
            encoding="utf-8",
        )
        (root / "util.py").write_text(
            "def auth_middleware():\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )

        settings = Settings(
            repo_default_root=str(root),
            repo_index_state_path=str(Path(tmp) / "index.json"),
            embedding_cache_path=str(Path(tmp) / "embeddings.sqlite3"),
            vector_db_path=str(Path(tmp) / "qdrant"),
            repo_incremental=False,
        )
        scanner = RepositoryScanner(settings)
        ast = TreeSitterAstEngine()
        chunker = AstAwareChunker()
        embeddings = FakeEmbeddingService(settings)
        vector_store = QdrantVectorStore(settings)
        retrieval = HybridRetrievalEngine(settings, embeddings, vector_store)
        graph = SymbolGraphEngine()

        scan_result = asyncio.run(scanner.scan(str(root)))
        all_chunks = []
        for repo_file in scan_result.files:
            parsed = ast.parse(repo_file)
            graph.upsert_parsed_file(parsed)
            all_chunks.extend(chunker.chunk(parsed))

        vectors = asyncio.run(embeddings.embed_chunks(all_chunks))
        asyncio.run(vector_store.upsert_chunks(all_chunks, vectors))
        retrieval.rebuild_lexical_index(all_chunks)

        context = asyncio.run(ContextBuilder(settings, retrieval, graph).build("auth middleware"))
        plan = asyncio.run(PlanningLayer(ContextBuilder(settings, retrieval, graph)).plan("auth middleware"))
        asyncio.run(vector_store.close())

        assert "util.py" in context.related_files
        assert context.hits
        assert plan.impacted_files
