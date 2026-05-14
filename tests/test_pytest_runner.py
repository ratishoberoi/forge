from __future__ import annotations

from pathlib import Path

import pytest

from backend.runtime.execution_policy import ExecutionPolicy
from backend.runtime.pytest_runner import PytestRunner


@pytest.mark.asyncio
async def test_pytest_runner_success(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert 1 == 1\n",
        encoding="utf-8",
    )
    runner = PytestRunner()

    result = await runner.run(tmp_path)

    assert result.success is True
    assert result.exit_code == 0
    assert "1 passed" in result.stdout


@pytest.mark.asyncio
async def test_pytest_runner_failure(tmp_path: Path) -> None:
    (tmp_path / "test_fail.py").write_text(
        "def test_fail():\n    assert 1 == 2\n",
        encoding="utf-8",
    )
    runner = PytestRunner()

    result = await runner.run(tmp_path)

    assert result.success is False
    assert result.exit_code == 1
    assert "FAILED" in result.stdout or "FAILED" in result.stderr


@pytest.mark.asyncio
async def test_pytest_runner_timeout(tmp_path: Path) -> None:
    (tmp_path / "test_sleep.py").write_text(
        "import time\n\ndef test_sleep():\n    time.sleep(1)\n",
        encoding="utf-8",
    )
    runner = PytestRunner(policy=ExecutionPolicy(timeout_seconds=0.05))

    result = await runner.run(tmp_path)

    assert result.success is False
    assert result.timed_out is True
    assert result.exit_code is None


@pytest.mark.asyncio
async def test_pytest_runner_accepts_custom_args(tmp_path: Path) -> None:
    (tmp_path / "test_a.py").write_text(
        "def test_a():\n    assert True\n",
        encoding="utf-8",
    )
    (tmp_path / "test_b.py").write_text(
        "def test_b():\n    assert True\n",
        encoding="utf-8",
    )
    runner = PytestRunner()

    result = await runner.run(tmp_path, "test_a.py")

    assert result.success is True
    assert "1 passed" in result.stdout
    assert "2 passed" not in result.stdout
