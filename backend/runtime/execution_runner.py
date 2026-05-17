from __future__ import annotations
import subprocess
import time
from backend.runtime.execution_result import ExecutionResult


class ExecutionRunnerError(Exception):
    """Raised when ExecutionRunner cannot run the command at all."""


class ExecutionRunner:
    """
    Executes repository validation commands via subprocess.
    Responsibilities:
    - run commands with captured output
    - record duration
    - support timeout
    - support environment injection
    - never raise on non-zero exit — caller inspects ExecutionResult
    """

    DEFAULT_TIMEOUT = 300

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def run(
        self,
        *,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Execute command and return ExecutionResult.
        Never raises on non-zero exit code — result.succeeded tells the story.
        Raises ExecutionRunnerError only if command cannot be started.
        """
        if not command:
            raise ExecutionRunnerError("command must not be empty.")

        effective_timeout = timeout or self.timeout
        start = time.monotonic()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env,
                timeout=effective_timeout,
            )
            duration = time.monotonic() - start

            return ExecutionResult(
                command=command,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=round(duration, 4),
            )

        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            return ExecutionResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"Command timed out after {effective_timeout}s.",
                duration_seconds=round(duration, 4),
            )
        except FileNotFoundError as exc:
            raise ExecutionRunnerError(
                f"Command not found: '{command[0]}'. Is it installed?"
            ) from exc
        except OSError as exc:
            raise ExecutionRunnerError(
                f"Failed to run command {command}: {exc}"
            ) from exc

    def run_script(
        self,
        script: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Run an inline Python script via `python -c`.
        Convenience wrapper for test/validation scripts.
        """
        return self.run(
            command=["python", "-c", script],
            cwd=cwd,
            timeout=timeout,
        )