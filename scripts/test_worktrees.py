import os
import subprocess
from pathlib import Path

import pytest

from backend.runtime.worktrees import WorktreeManager


@pytest.fixture()
def git_env(tmp_path: Path) -> dict[str, str]:
    return {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@test.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@test.com",
        "HOME": str(tmp_path),
        "PATH": os.environ["PATH"],
    }


@pytest.fixture()
def real_repo(
    tmp_path: Path,
    git_env: dict[str, str],
) -> Path:
    """
    Create isolated temporary git repo.
    """

    repo_path = tmp_path / "repo"

    subprocess.run(
        ["git", "init", str(repo_path)],
        check=True,
        env=git_env,
    )

    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=repo_path,
        check=True,
        env=git_env,
    )

    return repo_path


@pytest.fixture()
def manager(
    real_repo: Path,
    tmp_path: Path,
) -> WorktreeManager:
    worktree_root = tmp_path / "worktrees"

    return WorktreeManager(
        repo_root=str(real_repo),
        worktree_root=str(worktree_root),
    )


@pytest.mark.asyncio
async def test_create_worktree(
    manager: WorktreeManager,
):
    info = await manager.create_worktree(
        "coder-agent"
    )

    assert info.path.exists()
    assert info.branch_name.startswith(
        "coder-agent-"
    )
    assert len(info.workspace_id) == 8


@pytest.mark.asyncio
async def test_list_includes_created(
    manager: WorktreeManager,
):
    info = await manager.create_worktree(
        "coder-agent"
    )

    worktrees = await manager.list_worktrees()

    branch_names = [
        w.branch_name for w in worktrees
    ]

    assert info.branch_name in branch_names


@pytest.mark.asyncio
async def test_remove_worktree(
    manager: WorktreeManager,
):
    info = await manager.create_worktree(
        "coder-agent"
    )

    await manager.remove_worktree(info)

    assert not info.path.exists()

    worktrees = await manager.list_worktrees()

    branch_names = [
        w.branch_name for w in worktrees
    ]

    assert info.branch_name not in branch_names