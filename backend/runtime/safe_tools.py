from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.runtime.execution_result import ExecutionResult


class SafeToolError(RuntimeError):
    """Raised when a tool command violates workspace safety policy."""


@dataclass(slots=True)
class ToolPolicy:
    allow_dependency_install: bool = False
    allowed_roots: list[Path] = field(default_factory=list)
    allowed_commands: set[str] = field(
        default_factory=lambda: {
            "pytest",
            "python",
            "npm",
            "node",
            "mypy",
            "ruff",
            "flake8",
            "eslint",
            "git",
        }
    )


class SafeToolExecutor:
    """Runs approved repository tools inside an allowed workspace root."""

    def __init__(self, *, repo_root: str, policy: ToolPolicy | None = None, timeout: int = 300) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.policy = policy or ToolPolicy(allowed_roots=[self.repo_root])
        if not self.policy.allowed_roots:
            self.policy.allowed_roots = [self.repo_root]
        self.timeout = timeout

    def run(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        if not command:
            raise SafeToolError("command must not be empty.")
        executable = Path(command[0]).name
        if executable not in self.policy.allowed_commands:
            raise SafeToolError(f"command is not allowed: {executable}")
        if self._is_install_command(command) and not self.policy.allow_dependency_install:
            raise SafeToolError("dependency installation is disabled for this workspace.")
        workdir = Path(cwd).resolve() if cwd else self.repo_root
        if not any(workdir.is_relative_to(root.resolve()) for root in self.policy.allowed_roots):
            raise SafeToolError(f"cwd escapes allowed workspace roots: {workdir}")

        try:
            result = subprocess.run(
                command,
                cwd=workdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})},
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
            )
            return ExecutionResult(
                command=command,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=0.0,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout or self.timeout}s.",
            )
        except OSError as exc:
            raise SafeToolError(str(exc)) from exc

    @staticmethod
    def _is_install_command(command: list[str]) -> bool:
        return (
            len(command) >= 2
            and Path(command[0]).name in {"npm", "pip", "python"}
            and (
                command[1] in {"install", "i"}
                or command[1:3] == ["-m", "pip"]
                and len(command) >= 4
                and command[3] == "install"
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "allowed_roots": [str(root) for root in self.policy.allowed_roots],
            "allowed_commands": sorted(self.policy.allowed_commands),
            "allow_dependency_install": self.policy.allow_dependency_install,
        }
