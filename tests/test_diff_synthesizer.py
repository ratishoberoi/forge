import os
import subprocess
from pathlib import Path
import pytest
from backend.runtime.diff_synthesizer import DiffSynthesizer
from backend.runtime.gitops import GitOperations
from backend.runtime.output_parser import ParsedPatchOutput
from backend.runtime.patch_sandbox import PatchSandbox


# ── Fixtures ────────────────────────────────────────────────────────────────

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
def repo(tmp_path: Path, git_env: dict[str, str]) -> Path:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    subprocess.run(["git", "init", str(repo_path)], check=True, env=git_env)

    hello = repo_path / "hello.py"
    hello.write_text(
        "def hello(name):\n"
        "    return 'hello ' + name\n"
    )

    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, env=git_env)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_path, check=True, env=git_env)

    return repo_path


def make_parsed(files: dict[str, str], risk: str = "low") -> ParsedPatchOutput:
    return ParsedPatchOutput(
        summary="Add typing",
        reasoning="Improves clarity",
        risk=risk,
        files=files,
    )


def make_synthesizer(repo: Path) -> tuple[DiffSynthesizer, PatchSandbox]:
    sandbox = PatchSandbox(str(repo))
    gitops = GitOperations(str(repo))
    return DiffSynthesizer(sandbox=sandbox, gitops=gitops), sandbox


# ── synthesize_diff ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesize_diff_contains_changes(repo: Path):
    synthesizer, sandbox = make_synthesizer(repo)
    parsed = make_parsed({
        "hello.py": "def hello(name: str) -> str:\n    return 'hello ' + name\n"
    })

    await sandbox.materialize_patch(parsed)
    diff = await synthesizer.synthesize_diff()

    assert "diff --git" in diff
    assert "name: str" in diff


@pytest.mark.asyncio
async def test_synthesize_diff_empty_raises(repo: Path):
    """No changes in repo — diff must raise ValueError."""
    synthesizer, _ = make_synthesizer(repo)

    with pytest.raises(ValueError, match="No git diff generated"):
        await synthesizer.synthesize_diff()


# ── materialize_and_diff ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_materialize_and_diff_full_pipeline(repo: Path):
    synthesizer, _ = make_synthesizer(repo)
    parsed = make_parsed({
        "hello.py": "def hello(name: str) -> str:\n    return 'hello ' + name\n"
    })

    diff = await synthesizer.materialize_and_diff(parsed)

    assert "diff --git" in diff
    assert "name: str" in diff


@pytest.mark.asyncio
async def test_materialize_and_diff_empty_files_raises(repo: Path):
    synthesizer, _ = make_synthesizer(repo)
    parsed = make_parsed({})

    with pytest.raises(ValueError, match="empty"):
        await synthesizer.materialize_and_diff(parsed)


@pytest.mark.asyncio
async def test_materialize_and_diff_multiple_files(repo: Path):
    """Multiple files in parsed.files all appear in diff."""
    repo_path = repo

    # Create second file in initial commit
    utils = repo_path / "utils.py"
    utils.write_text("def util(): pass\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "add utils"], cwd=repo_path, check=True)

    synthesizer, _ = make_synthesizer(repo)
    parsed = make_parsed({
        "hello.py": "def hello(name: str) -> str:\n    return 'hello ' + name\n",
        "utils.py": "def util() -> None: pass\n",
    })

    diff = await synthesizer.materialize_and_diff(parsed)

    assert "hello.py" in diff
    assert "utils.py" in diff