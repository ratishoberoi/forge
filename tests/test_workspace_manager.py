from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from backend.config.settings import Settings
from backend.runtime.git_manager import GitManager
from backend.runtime.run_history import RunHistoryStore
from backend.runtime.workspace_manager import WorkspaceManager


def git_env(tmp_path: Path) -> dict[str, str]:
    return {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@test.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@test.com",
        "HOME": str(tmp_path),
        "PATH": os.environ["PATH"],
    }


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        repo_index_state_path=str(tmp_path / ".forge" / "index_state.json"),
        repo_incremental=False,
    )


def init_repo(path: Path, env: dict[str, str], *, package_name: str = "demo") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(
        f"[project]\nname='{package_name}'\n[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )
    (path / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (path / "tests").mkdir(exist_ok=True)
    (path / "tests" / "test_app.py").write_text(
        "from app import value\n\n"
        "def test_value():\n"
        "    assert value() == 1\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", str(path)], check=True, env=env)
    subprocess.run(["git", "add", "."], cwd=path, check=True, env=env)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, env=env)
    return path


def test_clone_branch_modify_test_and_commit(tmp_path: Path) -> None:
    env = git_env(tmp_path)
    source = init_repo(tmp_path / "source", env)
    manager = WorkspaceManager(
        registry_path=str(tmp_path / "registry.json"),
        workspace_root=str(tmp_path / "workspaces"),
        settings=make_settings(tmp_path),
        env=env,
    )

    record = manager.clone_repository(str(source), repository_name="source")
    git = GitManager(record.repository_path, env=env)
    branch = git.create_execution_branch("run-123")
    Path(record.repository_path, "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    Path(record.repository_path, "tests", "test_app.py").write_text(
        "from app import value\n\n"
        "def test_value():\n"
        "    assert value() == 2\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=record.repository_path,
        env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    commit = git.commit_all("forge: update value")

    assert branch == "forge/run-run-123"
    assert result.returncode == 0, result.stdout + result.stderr
    assert commit is not None
    assert git.status().is_dirty is False


def test_failed_run_can_rollback(tmp_path: Path) -> None:
    env = git_env(tmp_path)
    repo = init_repo(tmp_path / "repo", env)
    git = GitManager(str(repo), env=env)
    git.create_execution_branch("failed-run")
    (repo / "app.py").write_text("def value():\n    return 999\n", encoding="utf-8")

    assert git.status().is_dirty is True
    git.rollback("HEAD", clean_untracked=False)

    assert git.status().is_dirty is False
    assert "return 1" in (repo / "app.py").read_text(encoding="utf-8")


def test_multiple_repositories_keep_intelligence_isolated(tmp_path: Path) -> None:
    env = git_env(tmp_path)
    repo_a = init_repo(tmp_path / "repo_a", env, package_name="alpha")
    repo_b = init_repo(tmp_path / "repo_b", env, package_name="beta")
    (repo_b / "package.json").write_text('{"scripts":{"test":"echo ok"}}\n', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_b, check=True, env=env)
    subprocess.run(["git", "commit", "-m", "add package"], cwd=repo_b, check=True, env=env)
    manager = WorkspaceManager(
        registry_path=str(tmp_path / "registry.json"),
        workspace_root=str(tmp_path / "workspaces"),
        settings=make_settings(tmp_path),
        env=env,
    )
    record_a = manager.import_local_repository(str(repo_a), repository_name="alpha")
    record_b = manager.import_local_repository(str(repo_b), repository_name="beta")

    refreshed_a = asyncio.run(manager.refresh_intelligence(record_a.repository_id))
    refreshed_b = asyncio.run(manager.refresh_intelligence(record_b.repository_id))

    assert refreshed_a.repository_id != refreshed_b.repository_id
    assert refreshed_a.intelligence is not None
    assert refreshed_b.intelligence is not None
    assert refreshed_a.intelligence["scan"]["root"] == str(repo_a)
    assert refreshed_b.intelligence["scan"]["root"] == str(repo_b)
    assert refreshed_a.intelligence_signature != refreshed_b.intelligence_signature
    assert manager.get_repository(record_a.repository_id).intelligence["scan"]["root"] == str(repo_a)
    assert manager.get_repository(record_b.repository_id).intelligence["scan"]["root"] == str(repo_b)


def test_run_history_replay_persists_telemetry(tmp_path: Path) -> None:
    store = RunHistoryStore(str(tmp_path / "runs.json"))
    store.record_started(
        run_id="run-1",
        objective="Fix tests",
        repository_id="repo-1",
        repository_path=str(tmp_path),
        branch="forge/run-1",
    )
    store.record_completed(
        run_id="run-1",
        status="completed",
        result={"tests_passed": True},
        telemetry=["[PATCH_APPLIED] attempt=0 success=True", "[CONVERGED] tests_passed"],
        branch="forge/run-1",
        commit_sha="abc123",
    )

    replay = store.replay("run-1")

    assert store.get("run-1").commit_sha == "abc123"
    assert any(event["stage"] == "PATCH_APPLY" for event in replay)
    assert any(event["stage"] == "CONVERGENCE" for event in replay)
