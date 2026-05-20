from __future__ import annotations
import os
import subprocess
import time
from backend.runtime.runtime_process import RuntimeProcess

class RuntimeLaunchError(Exception):
    """Raised when runtime launch fails."""


class RuntimeLauncher:
    """
    Launches heavyweight vLLM runtimes.
    Designed for sequential one-GPU orchestration.
    Responsibilities:
    - build vLLM server launch command
    - redirect stdout/stderr to role-specific log files
    - mark process as launched
    - optional startup wait before returning
    """

    DEFAULT_STARTUP_WAIT = 0.5
    DEFAULT_LOG_DIR = "~/Forge/runtime_logs"
    DEFAULT_MAX_MODEL_LEN = 8192
    DEFAULT_GPU_MEMORY_UTILIZATION = "0.90"
    DEFAULT_MAX_NUM_SEQS = 64

    def __init__(
        self,
        startup_wait: float = DEFAULT_STARTUP_WAIT,
        log_dir: str = DEFAULT_LOG_DIR,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: str = DEFAULT_GPU_MEMORY_UTILIZATION,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        extra_args: list[str] | None = None,
    ) -> None:
        self.startup_wait = startup_wait
        self.log_dir = os.path.expanduser(log_dir)
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.extra_args = extra_args or []

    def launch(self, process: RuntimeProcess) -> RuntimeProcess:
        """
        Launch vLLM server for the given process.
        Logs stdout/stderr to role-specific files under log_dir.
        Marks process as launched on success.
        """
        if process.is_running:
            raise RuntimeLaunchError(
                f"Runtime already running for role='{process.role}' "
                f"(pid={process.pid})."
            )

        command = self._build_command(process)
        stdout_path, stderr_path = self._log_paths(process)

        os.makedirs(self.log_dir, exist_ok=True)

        try:
            with open(stdout_path, "ab") as stdout_file, \
                 open(stderr_path, "ab") as stderr_file:
                proc = subprocess.Popen(
                    command,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
        except FileNotFoundError as exc:
            raise RuntimeLaunchError(
                f"Command not found: '{command[0]}'. Is vLLM installed?"
            ) from exc
        except Exception as exc:
            raise RuntimeLaunchError(
                f"Failed to launch runtime for role='{process.role}': {exc}"
            ) from exc

        process.mark_launched(
            proc.pid,
            os.getpgid(proc.pid),
        )

        if self.startup_wait > 0:
            time.sleep(self.startup_wait)

        return_code = proc.poll()
        if return_code is not None:
            process.mark_stopped()
            raise RuntimeLaunchError(
                f"Runtime for role='{process.role}' exited during startup "
                f"with code {return_code}. See {stderr_path}."
            )

        return process

    def _build_command(self, process: RuntimeProcess) -> list[str]:
        """Build vLLM OpenAI-compatible server launch command."""
        command = [
            "python", "-m",
            "vllm.entrypoints.openai.api_server",
            "--model", os.path.expanduser(process.model_path),
            "--served-model-name", process.model_name,
            "--port", str(process.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", self.gpu_memory_utilization,
            "--max-num-seqs", str(self.max_num_seqs),
        ]
        command.extend(self.extra_args)
        return command

    def _log_paths(self, process: RuntimeProcess) -> tuple[str, str]:
        """Return (stdout_path, stderr_path) for this process role."""
        role = process.role.lower()
        return (
            os.path.join(self.log_dir, f"{role}_stdout.log"),
            os.path.join(self.log_dir, f"{role}_stderr.log"),
        )
