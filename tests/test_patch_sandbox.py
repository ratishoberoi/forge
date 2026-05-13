from pathlib import Path
import pytest
from backend.runtime.output_parser import ParsedPatchOutput
from backend.runtime.patch_sandbox import PatchSandbox


def make_parsed(files: dict[str, str], risk: str = "low") -> ParsedPatchOutput:
    return ParsedPatchOutput(
        summary="Add typing",
        reasoning="Improves clarity",
        risk=risk,
        files=files,
    )


# ── materialize_patch ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_materialize_patch_writes_file(tmp_path: Path):
    sandbox = PatchSandbox(str(tmp_path))
    parsed = make_parsed({"hello.py": "def hello(name: str) -> str:\n    return 'hello ' + name\n"})

    written = await sandbox.materialize_patch(parsed)

    assert len(written) == 1
    file_path = tmp_path / "hello.py"
    assert file_path.exists()
    assert "name: str" in file_path.read_text()


@pytest.mark.asyncio
async def test_materialize_patch_returns_resolved_paths(tmp_path: Path):
    sandbox = PatchSandbox(str(tmp_path))
    parsed = make_parsed({"hello.py": "content"})

    written = await sandbox.materialize_patch(parsed)

    assert written[0] == (tmp_path / "hello.py").resolve()


@pytest.mark.asyncio
async def test_materialize_patch_multiple_files(tmp_path: Path):
    sandbox = PatchSandbox(str(tmp_path))
    parsed = make_parsed({
        "hello.py": "def hello(): pass",
        "utils.py": "def util(): pass",
    })

    written = await sandbox.materialize_patch(parsed)

    assert len(written) == 2
    assert (tmp_path / "hello.py").exists()
    assert (tmp_path / "utils.py").exists()


@pytest.mark.asyncio
async def test_materialize_patch_empty_files_raises(tmp_path: Path):
    sandbox = PatchSandbox(str(tmp_path))
    parsed = make_parsed({})

    with pytest.raises(ValueError, match="empty"):
        await sandbox.materialize_patch(parsed)


# ── path traversal ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prevent_path_escape(tmp_path: Path):
    sandbox = PatchSandbox(str(tmp_path))
    parsed = make_parsed({"../evil.py": "x"})

    with pytest.raises(ValueError, match="Path traversal"):
        await sandbox.materialize_patch(parsed)


@pytest.mark.asyncio
async def test_prevent_deep_path_escape(tmp_path: Path):
    sandbox = PatchSandbox(str(tmp_path))
    parsed = make_parsed({"../../etc/passwd": "root:x:0:0"})

    with pytest.raises(ValueError, match="Path traversal"):
        await sandbox.materialize_patch(parsed)


# ── is_within_sandbox ───────────────────────────────────────────────────────

def test_is_within_sandbox_true(tmp_path: Path):
    sandbox = PatchSandbox(str(tmp_path))
    assert sandbox.is_within_sandbox(tmp_path / "hello.py") is True


def test_is_within_sandbox_false(tmp_path: Path):
    sandbox = PatchSandbox(str(tmp_path))
    assert sandbox.is_within_sandbox("/etc/passwd") is False


# ── list_materialized ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_materialized(tmp_path: Path):
    sandbox = PatchSandbox(str(tmp_path))
    parsed = make_parsed({"hello.py": "code", "utils.py": "code"})
    await sandbox.materialize_patch(parsed)

    listed = sandbox.list_materialized()
    names = [p.name for p in listed]

    assert "hello.py" in names
    assert "utils.py" in names