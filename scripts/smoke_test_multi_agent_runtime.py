"""Smoke test for the multi-agent runtime foundation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from backend.config.settings import Settings
from backend.runtime import MultiAgentRuntime
from backend.runtime.messages import TaskRequestPayload
from backend.runtime.tasks import Task, TaskPriority


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
            repo_index_state_path=str(Path(tmp) / "runtime-index.json"),
            embedding_cache_path=str(Path(tmp) / "runtime-embeddings.sqlite3"),
            vector_db_path=str(Path(tmp) / "runtime-qdrant"),
            repo_incremental=False,
            runtime_max_concurrency=2,
        )
        runtime = MultiAgentRuntime(settings)
        runtime.register_default_mock_agents()

        async def start_runtime() -> str:
            await runtime.start()
            return "runtime started"

        async def index_repo() -> str:
            stats = await runtime.repo_intelligence.index_repository(sample_root)
            return f"indexed {stats.files_indexed} files"

        async def submit_plan_task() -> str:
            task = Task(
                title="Plan auth middleware improvement",
                capability="planning",
                priority=TaskPriority.HIGH,
                payload=TaskRequestPayload(
                    objective="add caching to auth middleware",
                    target_files=["app.py", "util.py"],
                ),
            )
            task_id = await runtime.orchestrator.submit(task)
            completed = await runtime.orchestrator.wait_for_task(task_id)
            return completed.status.value

        async def submit_patch_task() -> str:
            task = Task(
                title="Propose integration patch",
                capability="patching",
                payload=TaskRequestPayload(
                    objective="wire runtime orchestration into backend startup",
                    target_files=["backend/app.py"],
                ),
            )
            task_id = await runtime.orchestrator.submit(task)
            completed = await runtime.orchestrator.wait_for_task(task_id)
            return completed.status.value

        async def diagnostics() -> str:
            snapshot = runtime.orchestrator.diagnostics()
            return f"active={snapshot.concurrency.active_tasks} completed={snapshot.concurrency.completed_tasks}"

        steps = [
            ("start runtime", start_runtime),
            ("index repository", index_repo),
            ("submit planning task", submit_plan_task),
            ("submit patch task", submit_patch_task),
            ("orchestration diagnostics", diagnostics),
        ]

        failed = False
        try:
            for name, fn in steps:
                ok, detail, ms = await run_step(name, fn)
                print(f"[{'PASS' if ok else 'FAIL'}] {name:<26} {detail} ({ms} ms)")
                if not ok:
                    failed = True
                    break
        finally:
            await runtime.stop()
        return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.repo_root)))
