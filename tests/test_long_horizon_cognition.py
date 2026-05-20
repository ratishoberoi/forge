from __future__ import annotations

import asyncio
from pathlib import Path

from backend.config.settings import Settings
from backend.runtime.architecture_memory import ArchitectureMemory
from backend.runtime.context_compressor import ContextCompressor
from backend.runtime.repository_execution_engine import RepositoryExecutionEngine
from backend.runtime.task_planner import TaskPlanner


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        repo_index_state_path=str(tmp_path / ".forge" / "index_state.json"),
        repo_incremental=False,
    )


def test_task_decomposition_for_oauth_login(tmp_path: Path) -> None:
    _write_auth_repo(tmp_path)
    memory = ArchitectureMemory(str(tmp_path / "memory.json"))
    engine = RepositoryExecutionEngine(
        repo_root=str(tmp_path),
        settings=make_settings(tmp_path),
        architecture_memory=memory,
    )

    prep = asyncio.run(engine.prepare("Add OAuth login"))

    assert prep.task_plan is not None
    goals = [task.goal for task in prep.task_plan.tasks]
    assert len(goals) >= 4
    assert any("authentication" in goal.lower() for goal in goals)
    assert prep.execution_graph is not None
    assert any(step.kind == "patch" for step in prep.execution_graph.steps)
    assert any(step.kind == "tests" for step in prep.execution_graph.steps)


def test_cross_file_reasoning_expands_import_neighbors(tmp_path: Path) -> None:
    _write_auth_repo(tmp_path)
    engine = RepositoryExecutionEngine(
        repo_root=str(tmp_path),
        settings=make_settings(tmp_path),
        architecture_memory=ArchitectureMemory(str(tmp_path / "memory.json")),
    )

    prep = asyncio.run(engine.prepare("Refactor authentication system"))

    assert "app/routes.py" in prep.context.dependency_graph
    assert "app/services/auth.py" in prep.context.relevant_files
    assert "app/models/user.py" in prep.context.relevant_files
    assert "tests/test_auth.py" in prep.context.related_tests


def test_context_compression_limits_large_repository_context(tmp_path: Path) -> None:
    _write_auth_repo(tmp_path)
    for index in range(40):
        path = tmp_path / "app" / f"module_{index}.py"
        path.write_text(
            "\n".join(f"def function_{line}():\n    return {line}" for line in range(40)),
            encoding="utf-8",
        )
    engine = RepositoryExecutionEngine(
        repo_root=str(tmp_path),
        settings=make_settings(tmp_path),
        architecture_memory=ArchitectureMemory(str(tmp_path / "memory.json")),
    )

    prep = asyncio.run(engine.prepare("Add OAuth login"))

    assert prep.compressed_context is not None
    assert prep.compressed_context.token_estimate <= ContextCompressor(max_tokens=6000).budget.max_tokens
    assert len(prep.compressed_context.file_summaries) <= 18


def test_architecture_memory_reused_and_updated_across_runs(tmp_path: Path) -> None:
    _write_auth_repo(tmp_path)
    memory = ArchitectureMemory(str(tmp_path / "memory.json"))
    engine = RepositoryExecutionEngine(
        repo_root=str(tmp_path),
        settings=make_settings(tmp_path),
        architecture_memory=memory,
    )
    first = asyncio.run(engine.prepare("Add OAuth login"))
    memory.record_outcome(
        repository_path=str(tmp_path),
        modified_files=["app/services/auth.py"],
        failure_patterns=["AssertionError"],
    )
    second = asyncio.run(engine.prepare("Add OAuth login"))

    assert first.architecture_memory is not None
    assert second.architecture_memory is not None
    assert "app/services/auth.py" in second.architecture_memory.previously_modified_files
    assert "AssertionError" in second.architecture_memory.recurring_failure_patterns


def test_task_planner_handles_architecture_migration(tmp_path: Path) -> None:
    _write_auth_repo(tmp_path)
    engine = RepositoryExecutionEngine(
        repo_root=str(tmp_path),
        settings=make_settings(tmp_path),
        architecture_memory=ArchitectureMemory(str(tmp_path / "memory.json")),
    )
    prep = asyncio.run(engine.prepare("Convert Flask to FastAPI"))
    plan = TaskPlanner().plan(objective="Convert Flask to FastAPI", preparation=prep, memory=prep.architecture_memory)

    assert len(plan.tasks) >= 4
    assert any("FastAPI" in task.goal for task in plan.tasks)
    assert plan.tasks[1].dependencies == ["task-1"]


def _write_auth_repo(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "[project]\nname='auth-app'\n[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )
    (root / "app").mkdir()
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "routes.py").write_text(
        "from app.services.auth import authenticate\n\n"
        "def login(email: str, password: str):\n"
        "    return authenticate(email, password)\n",
        encoding="utf-8",
    )
    (root / "app" / "services").mkdir()
    (root / "app" / "services" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "services" / "auth.py").write_text(
        "from app.models.user import User\n\n"
        "def authenticate(email: str, password: str):\n"
        "    return User(email=email)\n",
        encoding="utf-8",
    )
    (root / "app" / "models").mkdir()
    (root / "app" / "models" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "models" / "user.py").write_text(
        "class User:\n"
        "    def __init__(self, email: str):\n"
        "        self.email = email\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_auth.py").write_text(
        "from app.routes import login\n\n"
        "def test_login():\n"
        "    assert login('a@example.com', 'pw').email == 'a@example.com'\n",
        encoding="utf-8",
    )
