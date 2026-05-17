import os
import subprocess
from pathlib import Path
import pytest
from backend.runtime.git_diff import GitDiff, GitDiffError
from backend.runtime.patch_writer import PatchWriter, PatchWriterError, PatchResult
from backend.runtime.repo_workspace import RepositoryWorkspace, WorkspaceError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def workspace(tmp_path: Path) -> RepositoryWorkspace:
    return RepositoryWorkspace(str(tmp_path))


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@test.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@test.com",
        "HOME": str(tmp_path),
        "PATH": os.environ["PATH"],
    }
    subprocess.run(["git", "init", str(tmp_path)], check=True, env=env)
    hello = tmp_path / "hello.py"
    hello.write_text("def hello(): pass\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path, check=True, env=env,
    )
    return tmp_path


# ── RepositoryWorkspace ───────────────────────────────────────────────────────

def test_workspace_write_and_read(workspace: RepositoryWorkspace):
    workspace.write(relative_path="a/test.txt", content="hello")
    assert workspace.read("a/test.txt") == "hello"


def test_workspace_write_returns_path(workspace: RepositoryWorkspace):
    path = workspace.write(relative_path="out.txt", content="data")
    assert path.exists()
    assert path.suffix == ".txt"


def test_workspace_read_missing_raises(workspace: RepositoryWorkspace):
    with pytest.raises(WorkspaceError, match="not found"):
        workspace.read("nonexistent.txt")


def test_workspace_path_escape_raises(workspace: RepositoryWorkspace):
    with pytest.raises(ValueError, match="Path traversal"):
        workspace.write(relative_path="../../escape.txt", content="bad")


def test_workspace_path_escape_read_raises(workspace: RepositoryWorkspace):
    with pytest.raises(ValueError, match="Path traversal"):
        workspace.read("../../etc/passwd")


def test_workspace_exists_true(workspace: RepositoryWorkspace):
    workspace.write(relative_path="exists.txt", content="data")
    assert workspace.exists("exists.txt") is True


def test_workspace_exists_false(workspace: RepositoryWorkspace):
    assert workspace.exists("nonexistent.txt") is False


def test_workspace_exists_returns_false_on_traversal(workspace: RepositoryWorkspace):
    assert workspace.exists("../../escape.txt") is False


def test_workspace_delete(workspace: RepositoryWorkspace):
    workspace.write(relative_path="del.txt", content="data")
    assert workspace.delete("del.txt") is True
    assert workspace.exists("del.txt") is False


def test_workspace_delete_missing_returns_false(workspace: RepositoryWorkspace):
    assert workspace.delete("nonexistent.txt") is False


def test_workspace_list_files(workspace: RepositoryWorkspace):
    workspace.write(relative_path="a.txt", content="1")
    workspace.write(relative_path="b.txt", content="2")
    files = workspace.list_files("*.txt")
    names = [p.name for p in files]
    assert "a.txt" in names
    assert "b.txt" in names


def test_workspace_relative_to_root(workspace: RepositoryWorkspace, tmp_path: Path):
    path = workspace.write(relative_path="sub/file.txt", content="x")
    rel = workspace.relative_to_root(path)
    assert rel == "sub/file.txt"


def test_workspace_nonexistent_root_raises(tmp_path: Path):
    with pytest.raises(WorkspaceError, match="does not exist"):
        RepositoryWorkspace(str(tmp_path / "nonexistent"))


# ── PatchWriter ───────────────────────────────────────────────────────────────

def test_patch_writer_apply(workspace: RepositoryWorkspace):
    writer = PatchWriter(workspace)
    result = writer.apply(file_path="b/code.py", new_content="print(1)")
    assert result.success is True
    assert workspace.read("b/code.py") == "print(1)"


def test_patch_writer_apply_returns_patch_result(workspace: RepositoryWorkspace):
    writer = PatchWriter(workspace)
    result = writer.apply(file_path="out.py", new_content="x = 1")
    assert isinstance(result, PatchResult)
    assert result.resolved_path is not None


def test_patch_writer_blank_content_raises(workspace: RepositoryWorkspace):
    writer = PatchWriter(workspace)
    with pytest.raises(PatchWriterError, match="blank"):
        writer.apply(file_path="out.py", new_content="   ")


def test_patch_writer_backup(workspace: RepositoryWorkspace):
    workspace.write(relative_path="file.py", content="original")
    writer = PatchWriter(workspace, backup=True)
    writer.apply(file_path="file.py", new_content="updated")

    assert workspace.read("file.py") == "updated"
    assert workspace.read("file.py.bak") == "original"


def test_patch_writer_apply_many(workspace: RepositoryWorkspace):
    writer = PatchWriter(workspace)
    results = writer.apply_many({
        "a.py": "x = 1",
        "b.py": "y = 2",
    })
    assert len(results) == 2
    assert all(r.success for r in results)
    assert workspace.read("a.py") == "x = 1"
    assert workspace.read("b.py") == "y = 2"


def test_patch_writer_apply_many_empty_raises(workspace: RepositoryWorkspace):
    writer = PatchWriter(workspace)
    with pytest.raises(PatchWriterError, match="empty"):
        writer.apply_many({})


def test_patch_writer_path_escape_returns_failed_result(workspace: RepositoryWorkspace):
    writer = PatchWriter(workspace)
    result = writer.apply(file_path="../../escape.py", new_content="bad")
    assert result.success is False
    assert result.error is not None


# ── GitDiff ───────────────────────────────────────────────────────────────────

def test_git_diff_callable(git_repo: Path):
    diff = GitDiff(repo_root=str(git_repo))
    result = diff.diff()
    assert isinstance(result, str)


def test_git_diff_empty_on_clean_repo(git_repo: Path):
    diff = GitDiff(repo_root=str(git_repo))
    assert diff.diff() == ""


def test_git_diff_shows_changes(git_repo: Path):
    (git_repo / "hello.py").write_text("def hello(name: str): pass\n")
    diff = GitDiff(repo_root=str(git_repo))
    result = diff.diff()
    assert "hello.py" in result or "diff" in result


def test_git_diff_staged(git_repo: Path):
    diff = GitDiff(repo_root=str(git_repo))
    result = diff.diff_staged()
    assert isinstance(result, str)


def test_git_diff_is_dirty_false_on_clean(git_repo: Path):
    diff = GitDiff(repo_root=str(git_repo))
    assert diff.is_dirty() is False


def test_git_diff_is_dirty_true_after_change(git_repo: Path):
    (git_repo / "hello.py").write_text("def hello(name: str): pass\n")
    diff = GitDiff(repo_root=str(git_repo))
    assert diff.is_dirty() is True


def test_git_diff_changed_files(git_repo: Path):
    (git_repo / "hello.py").write_text("def hello(name: str): pass\n")
    diff = GitDiff(repo_root=str(git_repo))
    files = diff.changed_files()
    assert "hello.py" in files


def test_git_diff_stat(git_repo: Path):
    (git_repo / "hello.py").write_text("def hello(name: str): pass\n")
    diff = GitDiff(repo_root=str(git_repo))
    stat = diff.stat()
    assert isinstance(stat, str)