"""Safe pytest execution inside an isolated workspace."""

from __future__ import annotations

from pathlib import Path
import sys

from backend.runtime.execution import ExecutionResult, ExecutionRunner
from backend.runtime.execution_policy import ExecutionPolicy


class PytestRunner:
    """Runs `pytest -q` within an isolated current working directory."""

    def __init__(
        self,
        execution_runner: ExecutionRunner | None = None,
        policy: ExecutionPolicy | None = None,
    ) -> None:
        self._policy = policy or ExecutionPolicy()
        self._runner = execution_runner or ExecutionRunner(self._policy)

    async def run(
        self,
        cwd: str | Path,
        *pytest_args: str,
        policy: ExecutionPolicy | None = None,
    ) -> ExecutionResult:
        return await self._runner.run(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *pytest_args,
            cwd=cwd,
            policy=policy or self._policy,
        )
