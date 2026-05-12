"""Benchmark orchestration throughput and task latency."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from backend.config.settings import Settings
from backend.runtime import MultiAgentRuntime
from backend.runtime.messages import TaskRequestPayload
from backend.runtime.tasks import Task


def build_sample_repo(root: Path) -> str:
    repo_root = root / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "main.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    return str(repo_root)


async def main(task_count: int) -> None:
    with TemporaryDirectory() as tmp:
        repo_root = build_sample_repo(Path(tmp))
        settings = Settings(
            repo_default_root=repo_root,
            repo_index_state_path=str(Path(tmp) / "bench-index.json"),
            embedding_cache_path=str(Path(tmp) / "bench-embeddings.sqlite3"),
            vector_db_path=str(Path(tmp) / "bench-qdrant"),
            repo_incremental=False,
            runtime_max_concurrency=4,
        )
        runtime = MultiAgentRuntime(settings)
        runtime.register_default_mock_agents()
        await runtime.start()
        try:
            await runtime.repo_intelligence.index_repository(repo_root)
            started = perf_counter()
            task_ids = []
            for index in range(task_count):
                task = Task(
                    title=f"Plan task {index}",
                    capability="planning",
                    payload=TaskRequestPayload(
                        objective=f"inspect task {index}",
                        target_files=["main.py"],
                    ),
                )
                task_ids.append(await runtime.orchestrator.submit(task))
            for task_id in task_ids:
                await runtime.orchestrator.wait_for_task(task_id)
            total_ms = (perf_counter() - started) * 1000
            diagnostics = runtime.orchestrator.diagnostics()
            print(
                {
                    "task_count": task_count,
                    "total_ms": round(total_ms, 3),
                    "tasks_per_second": round(task_count / max(total_ms / 1000, 0.001), 3),
                    "completed_tasks": diagnostics.concurrency.completed_tasks,
                    "avg_agent_latency_ms": [
                        metric.average_latency_ms for metric in diagnostics.agent_metrics
                    ],
                }
            )
        finally:
            await runtime.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-count", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args.task_count))
