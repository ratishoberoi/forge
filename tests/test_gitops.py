import os
import subprocess
from pathlib import Path

import pytest

from backend.runtime.gitops import GitOperations


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
def repo(
    tmp_path: Path,
    git_env: dict[str, str],
) -> Path:
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
def gitops(repo: Path) -> GitOperations:
    return GitOperations(
        repo_path=str(repo)
    )


@pytest.mark.asyncio
async def test_current_branch(
    gitops: GitOperations,
):
    branch = await gitops.current_branch()

    assert branch in {
        "main",
        "master",
    }


@pytest.mark.asyncio
async def test_changed_files(
    repo: Path,
    gitops: GitOperations,
    git_env: dict[str, str],
):
    tracked = repo / "hello.py"

    tracked.write_text(
        "print('hello')\n"
    )

    subprocess.run(
        ["git", "add", "."],
        cwd=repo,
        check=True,
        env=git_env,
    )

    subprocess.run(
        ["git", "commit", "-m", "add file"],
        cwd=repo,
        check=True,
        env=git_env,
    )

    tracked.write_text(
        "print('modified')\n"
    )

    changed = await gitops.changed_files()

    assert "hello.py" in changed


@pytest.mark.asyncio
async def test_untracked_files(
    repo: Path,
    gitops: GitOperations,
):
    test_file = repo / "new_file.py"

    test_file.write_text(
        "print('new')\n"
    )

    untracked = await gitops.untracked_files()

    assert "new_file.py" in untracked


@pytest.mark.asyncio
async def test_diff_generation(
    repo: Path,
    gitops: GitOperations,
    git_env: dict[str, str],
):
    tracked = repo / "tracked.py"

    tracked.write_text(
        "print('v1')\n"
    )

    subprocess.run(
        ["git", "add", "."],
        cwd=repo,
        check=True,
        env=git_env,
    )

    subprocess.run(
        ["git", "commit", "-m", "add tracked"],
        cwd=repo,
        check=True,
        env=git_env,
    )

    tracked.write_text(
        "print('v2')\n"
    )

    diff = await gitops.get_diff()

    assert "v2" in diff


@pytest.mark.asyncio
async def test_repo_status(
    repo: Path,
    gitops: GitOperations,
    git_env: dict[str, str],
):
    untracked = repo / "new.py"

    untracked.write_text(
        "x = 1\n"
    )

    tracked = repo / "tracked.py"

    tracked.write_text(
        "print('v1')\n"
    )

    subprocess.run(
        ["git", "add", "tracked.py"],
        cwd=repo,
        check=True,
        env=git_env,
    )

    subprocess.run(
        ["git", "commit", "-m", "add tracked"],
        cwd=repo,
        check=True,
        env=git_env,
    )

    tracked.write_text(
        "print('v2')\n"
    )

    status = await gitops.repo_status()

    assert status.branch in {
        "main",
        "master",
    }

    assert (
        "tracked.py"
        in status.modified_files
    )

    assert (
        "new.py"
        in status.untracked_files
    )