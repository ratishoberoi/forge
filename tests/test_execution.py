from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend.runtime.execution import ExecutionRunner
from backend.runtime.execution_policy import ExecutionPolicy


@pytest.mark.asyncio
async def test_execution_runner_successful_execution(tmp_path: Path) -> None:
    runner = ExecutionRunner()

    result = await runner.run(
        sys.executable,
        "-c",
        "print('forge-ok')",
        cwd=tmp_path,
    )

    assert result.success is True
    assert result.exit_code == 0
    assert result.stdout.strip() == "forge-ok"
    assert result.stderr == ""
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_execution_runner_failing_execution_captures_stderr(tmp_path: Path) -> None:
    runner = ExecutionRunner()

    result = await runner.run(
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('forge-error\\n'); raise SystemExit(3)",
        cwd=tmp_path,
    )

    assert result.success is False
    assert result.exit_code == 3
    assert result.stdout == ""
    assert "forge-error" in result.stderr
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_execution_runner_timeout_handling(tmp_path: Path) -> None:
    runner = ExecutionRunner(ExecutionPolicy(timeout_seconds=0.05))

    result = await runner.run(
        sys.executable,
        "-c",
        "import time; time.sleep(1)",
        cwd=tmp_path,
    )

    assert result.success is False
    assert result.exit_code is None
    assert result.timed_out is True
    assert "timed out" in result.stderr


@pytest.mark.asyncio
async def test_execution_runner_respects_max_output_chars(tmp_path: Path) -> None:
    runner = ExecutionRunner(ExecutionPolicy(max_output_chars=5))

    result = await runner.run(
        sys.executable,
        "-c",
        "print('1234567890')",
        cwd=tmp_path,
    )

    assert result.stdout == "12345"


@pytest.mark.asyncio
async def test_execution_runner_uses_isolated_cwd(tmp_path: Path) -> None:
    runner = ExecutionRunner()

    result = await runner.run(
        sys.executable,
        "-c",
        "from pathlib import Path; print(Path.cwd().name)",
        cwd=tmp_path,
    )

    assert result.success is True
    assert result.stdout.strip() == tmp_path.name


@pytest.mark.asyncio
async def test_execution_runner_missing_executable_is_safe() -> None:
    runner = ExecutionRunner()

    result = await runner.run("definitely-not-a-real-executable")

    assert result.success is False
    assert result.exit_code == 127
    assert "Executable not found" in result.stderr


@pytest.mark.asyncio
async def test_execution_runner_no_shell_interpretation(tmp_path: Path) -> None:
    runner = ExecutionRunner()

    result = await runner.run(
        sys.executable,
        "-c",
        "import sys; print(sys.argv[1])",
        "unsafe;echo nope",
        cwd=tmp_path,
    )

    assert result.success is True
    assert result.stdout.strip() == "unsafe;echo nope"

