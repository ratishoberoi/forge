"""Example execution flow for the multi-agent runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.config.settings import Settings
from backend.runtime import MultiAgentRuntime
from backend.runtime.messages import TaskRequestPayload
from backend.runtime.tasks import Task, TaskPriority


def build_sample_repo(root: Path) -> str:
    repo_root = root / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "main.py").write_text(
        "from auth import middleware\n\n"
        "def create_app():\n"
        "    return middleware()\n",
        encoding="utf-8",
    )
    (repo_root / "auth.py").write_text(
        "def middleware():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    return str(repo_root)


async def main() -> None:
    with TemporaryDirectory() as tmp:
        repo_root = build_sample_repo(Path(tmp))
        settings = Settings(
            repo_default_root=repo_root,
            repo_index_state_path=str(Path(tmp) / "demo-index.json"),
            embedding_cache_path=str(Path(tmp) / "demo-embeddings.sqlite3"),
            vector_db_path=str(Path(tmp) / "demo-qdrant"),
            repo_incremental=False,
        )
        runtime = MultiAgentRuntime(settings)
        runtime.register_default_mock_agents()
        await runtime.start()
        try:
            await runtime.repo_intelligence.index_repository(repo_root)
            plan_task = Task(
                title="Plan middleware caching",
                capability="planning",
                priority=TaskPriority.HIGH,
                payload=TaskRequestPayload(
                    objective="add caching to middleware",
                    target_files=["main.py", "auth.py"],
                ),
            )
            patch_task = Task(
                title="Draft patch",
                capability="patching",
                dependency_ids=[plan_task.id],
                payload=TaskRequestPayload(
                    objective="draft patch for middleware caching",
                    target_files=["auth.py"],
                ),
            )
            await runtime.orchestrator.submit(plan_task)
            await runtime.orchestrator.submit(patch_task)
            plan_result = await runtime.orchestrator.wait_for_task(plan_task.id)
            patch_result = await runtime.orchestrator.wait_for_task(patch_task.id)
            print({"plan_status": plan_result.status, "patch_status": patch_result.status})
            print(runtime.orchestrator.diagnostics().model_dump())
        finally:
            await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
