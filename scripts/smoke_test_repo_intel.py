"""Human-readable smoke test for repository intelligence public interfaces."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from backend.config.settings import Settings
from backend.repointel.api import (
    ContextBuilder,
    EmbeddingService,
    PlanningLayer,
    RepositoryIntelligenceEngine,
    TreeSitterParser,
)


async def run_step(name: str, fn) -> tuple[bool, str, float]:
    started = perf_counter()
    try:
        detail = await fn()
        return True, detail, round((perf_counter() - started) * 1000, 3)
    except Exception as exc:
        return False, str(exc), round((perf_counter() - started) * 1000, 3)


def build_sample_repo(root: Path) -> str:
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


async def main(repo_root: str | None) -> int:
    with TemporaryDirectory() as tmp:
        sample_root = repo_root or build_sample_repo(Path(tmp))
        settings = Settings(
            repo_default_root=sample_root,
            repo_index_state_path=str(Path(tmp) / "index.json"),
            embedding_cache_path=str(Path(tmp) / "embeddings.sqlite3"),
            vector_db_path=str(Path(tmp) / "qdrant"),
            repo_incremental=False,
        )
        engine = RepositoryIntelligenceEngine(settings)
        parser = TreeSitterParser(settings)
        embeddings = EmbeddingService(settings)
        context_builder = engine.context_builder
        planner = engine.planner

        async def parser_init() -> str:
            validated = parser.validate_languages()
            return f"validated {len(validated)} languages"

        async def embedding_init() -> str:
            status = await embeddings.validate_startup()
            return f"{status['model_name']} dim={status['embedding_dimension']}"

        async def embedding_generation() -> str:
            vector = await embeddings.embed_text("auth middleware smoke test")
            return f"generated vector dim={len(vector)}"

        async def qdrant_connectivity() -> str:
            status = await engine.vector_store.healthcheck()
            return f"collections={status['collections']}"

        async def repository_indexing() -> str:
            diagnostics = await engine.index_repository_with_diagnostics(sample_root)
            return f"files={diagnostics.stats.files_indexed} chunks={diagnostics.stats.chunks_indexed}"

        async def retrieval() -> str:
            hits = await engine.retrieval.retrieve("auth middleware")
            return f"hits={len(hits)}"

        async def context_building() -> str:
            context = await context_builder.build("auth middleware")
            return f"files={len(context.related_files)} symbols={len(context.related_symbols)}"

        async def planning() -> str:
            plan = await planner.plan("auth middleware")
            return f"impacted_files={len(plan.impacted_files)} steps={len(plan.steps)}"

        steps = [
            ("parser init", parser_init),
            ("embedding init", embedding_init),
            ("embedding generation", embedding_generation),
            ("Qdrant connectivity", qdrant_connectivity),
            ("repository indexing", repository_indexing),
            ("retrieval", retrieval),
            ("context building", context_building),
            ("planning", planning),
        ]

        failed = False
        try:
            for name, fn in steps:
                ok, detail, ms = await run_step(name, fn)
                print(f"[{'PASS' if ok else 'FAIL'}] {name:<22} {detail} ({ms} ms)")
                if not ok:
                    failed = True
                    break
        finally:
            await engine.shutdown()
        return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.repo_root)))
