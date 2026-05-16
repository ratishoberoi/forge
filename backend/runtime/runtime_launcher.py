from __future__ import annotations
import subprocess
import time
from backend.runtime.runtime_process import RuntimeProcess


class RuntimeLaunchError(Exception):
    """Raised when a runtime process fails to launch."""


class RuntimeLauncher:
    """
    Spawns vLLM OpenAI-compatible inference servers.
    Responsibilities:
    - build launch command
    - spawn subprocess
    - update RuntimeProcess state
    - optionally wait for readiness
    """

    DEFAULT_STARTUP_WAIT = 0.0

    def __init__(
        self,
        startup_wait: float = DEFAULT_STARTUP_WAIT,
        extra_args: list[str] | None = None,
    ) -> None:
        self.startup_wait = startup_wait
        self.extra_args = extra_args or []

    def launch(self, process: RuntimeProcess) -> RuntimeProcess:
        """
        Spawn the inference server for the given process.
        Updates process.pid and process.active on success.
        """
        if process.is_running:
            raise RuntimeLaunchError(
                f"Process for role '{process.role}' is already running "
                f"(pid={process.pid})."
            )

        command = self._build_command(process)

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeLaunchError(
                f"Failed to launch runtime for role '{process.role}': {exc}"
            ) from exc

        process.mark_launched(proc.pid)

        if self.startup_wait > 0:
            time.sleep(self.startup_wait)

        return process

    def _build_command(self, process: RuntimeProcess) -> list[str]:
        """Build the vLLM server launch command."""
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model", process.model,
            "--served-model-name", process.role.lower(),
            "--port", str(process.port),
        ]
        command.extend(self.extra_args)
        return command