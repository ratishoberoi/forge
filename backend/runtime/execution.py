"""Safe subprocess execution utilities for isolated validation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from backend.runtime.execution_policy import ExecutionPolicy


@dataclass(slots=True)
class ExecutionResult:
    tool: str
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration: float
    timed_out: bool


class ExecutionRunner:
    """Runs subprocess commands without invoking a shell."""

    def __init__(self, policy: ExecutionPolicy | None = None) -> None:
        self._policy = policy or ExecutionPolicy()

    async def run(
        self,
        tool: str,
        *args: str,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        policy: ExecutionPolicy | None = None,
    ) -> ExecutionResult:
        if not tool or not tool.strip():
            raise ValueError("tool must be a non-empty executable name.")

        active_policy = policy or self._policy
        resolved_cwd = Path(cwd).resolve() if cwd is not None else None
        if resolved_cwd is not None and not resolved_cwd.exists():
            raise ValueError(f"cwd does not exist: {resolved_cwd}")

        started = time.perf_counter()
        stdout_pipe = asyncio.subprocess.PIPE if active_policy.capture_output else asyncio.subprocess.DEVNULL
        stderr_pipe = asyncio.subprocess.PIPE if active_policy.capture_output else asyncio.subprocess.DEVNULL

        try:
            process = await asyncio.create_subprocess_exec(
                tool,
                *args,
                cwd=str(resolved_cwd) if resolved_cwd is not None else None,
                env=dict(env) if env is not None else None,
                stdout=stdout_pipe,
                stderr=stderr_pipe,
            )
        except FileNotFoundError:
            duration = time.perf_counter() - started
            return ExecutionResult(
                tool=tool,
                success=False,
                exit_code=127,
                stdout="",
                stderr=f"Executable not found: {tool}",
                duration=round(duration, 6),
                timed_out=False,
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=active_policy.timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            duration = time.perf_counter() - started
            return ExecutionResult(
                tool=tool,
                success=False,
                exit_code=None,
                stdout="",
                stderr=f"Execution timed out after {active_policy.timeout_seconds} seconds.",
                duration=round(duration, 6),
                timed_out=True,
            )

        duration = time.perf_counter() - started
        stdout = self._normalize_output(stdout_bytes, active_policy)
        stderr = self._normalize_output(stderr_bytes, active_policy)
        return ExecutionResult(
            tool=tool,
            success=process.returncode == 0,
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            duration=round(duration, 6),
            timed_out=False,
        )

    @staticmethod
    def _normalize_output(data: bytes | None, policy: ExecutionPolicy) -> str:
        if not policy.capture_output or data is None:
            return ""
        text = data.decode("utf-8", errors="replace")
        if policy.max_output_chars == 0:
            return ""
        if len(text) <= policy.max_output_chars:
            return text
        return text[: policy.max_output_chars]
