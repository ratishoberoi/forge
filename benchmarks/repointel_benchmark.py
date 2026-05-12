"""Benchmark repository indexing and retrieval latency."""

from __future__ import annotations

import argparse
import asyncio
from time import perf_counter

from backend.config.settings import Settings
from backend.repointel.service import RepositoryIntelligenceEngine


async def main(repo_root: str, query: str) -> None:
    engine = RepositoryIntelligenceEngine(Settings())
    try:
        diagnostics = await engine.index_repository_with_diagnostics(repo_root)
        stats = diagnostics.stats
        indexing_ms = (
            diagnostics.scan_latency_ms
            + diagnostics.parsing_latency_ms
            + diagnostics.embedding_latency_ms
            + diagnostics.vector_latency_ms
        )

        retrieval_start = perf_counter()
        context = await engine.build_context(query)
        retrieval_ms = (perf_counter() - retrieval_start) * 1000

        print(
            {
                "files_scanned": stats.files_scanned,
                "files_indexed": stats.files_indexed,
                "symbols_extracted": stats.symbols_extracted,
                "chunks_indexed": stats.chunks_indexed,
                "indexing_ms": round(indexing_ms, 3),
                "files_per_second": round(stats.files_indexed / max(indexing_ms / 1000, 0.001), 3),
                "scan_latency_ms": diagnostics.scan_latency_ms,
                "parsing_latency_ms": diagnostics.parsing_latency_ms,
                "embedding_latency_ms": diagnostics.embedding_latency_ms,
                "vector_insertion_latency_ms": diagnostics.vector_latency_ms,
                "retrieval_ms": round(retrieval_ms, 3),
                "hits": len(context.hits),
            }
        )
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root")
    parser.add_argument("query")
    args = parser.parse_args()
    asyncio.run(main(args.repo_root, args.query))
