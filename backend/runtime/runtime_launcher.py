from __future__ import annotations
from collections.abc import Callable
from dataclasses import asdict, dataclass
import errno
import os
from pathlib import Path
import socket
import subprocess
import time
from backend.runtime.runtime_process import RuntimeProcess

class RuntimeLaunchError(Exception):
    """Raised when runtime launch fails."""


@dataclass(slots=True)
class RuntimeLaunchProfile:
    max_model_len: int
    gpu_memory_utilization: str
    max_num_seqs: int
    enforce_eager: bool = False
    model_path: str | None = None
    reason: str = "default"


@dataclass(slots=True)
class RuntimeMemorySnapshot:
    total_mb: int | None
    used_mb: int | None
    free_mb: int | None
    source: str = "unknown"
    error: str | None = None


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
    DEFAULT_STARTUP_FAILURE_WINDOW = 8.0

    def __init__(
        self,
        startup_wait: float = DEFAULT_STARTUP_WAIT,
        log_dir: str = DEFAULT_LOG_DIR,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: str = DEFAULT_GPU_MEMORY_UTILIZATION,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        extra_args: list[str] | None = None,
        startup_failure_window: float = DEFAULT_STARTUP_FAILURE_WINDOW,
        telemetry: Callable[[str], None] | None = None,
        diagnostics_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.startup_wait = startup_wait
        self.log_dir = os.path.expanduser(log_dir)
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.extra_args = extra_args or []
        self.startup_failure_window = max(startup_wait, startup_failure_window)
        self.telemetry = telemetry or print
        self.diagnostics_callback = diagnostics_callback
        self.last_diagnostics: dict[str, object] = {}

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

        os.makedirs(self.log_dir, exist_ok=True)
        requested_port = process.port
        last_error: Exception | None = None
        profiles = self._launch_profiles(process)
        for profile_index, profile in enumerate(profiles, start=1):
            precheck = self._runtime_precheck(process, profile, profile_index, len(profiles))
            if not precheck["fits"]:
                last_error = RuntimeLaunchError(str(precheck["failure_reason"]))
                if profile_index < len(profiles):
                    self._emit(
                        f"[MODEL_FALLBACK] {process.role} profile={profile.reason} "
                        f"reason={precheck['failure_reason']}"
                    )
                    continue
                self._emit(f"[MODEL_FAILED] {process.role} {precheck['failure_reason']}")
                raise last_error

            for attempt in range(2):
                if self._port_in_use(process.port):
                    self._reassign_port(process, requested_port)

                command = self._build_command(process, profile)
                stdout_path, stderr_path = self._log_paths(process)
                self._publish_diagnostics(
                    process,
                    profile,
                    precheck,
                    status="launching",
                    launch_args=command,
                    failure_reason=None,
                )
                self._emit(
                    f"[MODEL_LOAD] {process.role} model={process.model_name} "
                    f"profile={profile.reason} port={process.port}"
                )

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
                except OSError as exc:
                    last_error = exc
                    if exc.errno == errno.EADDRINUSE and attempt == 0:
                        self._reassign_port(process, requested_port)
                        continue
                    raise RuntimeLaunchError(
                        f"Failed to launch runtime for role='{process.role}': {exc}"
                    ) from exc
                except Exception as exc:
                    raise RuntimeLaunchError(
                        f"Failed to launch runtime for role='{process.role}': {exc}"
                    ) from exc

                process.mark_launched(
                    proc.pid,
                    os.getpgid(proc.pid),
                )
                process.metadata = {
                    **process.metadata,
                    "_popen": proc,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                    "runtime_diagnostics": self.last_diagnostics,
                }

                return_code = self._wait_for_early_failure(proc)
                if return_code is None:
                    self._publish_diagnostics(
                        process,
                        profile,
                        precheck,
                        status="started",
                        launch_args=command,
                        failure_reason=None,
                    )
                    return process

                process.mark_stopped()
                stderr_tail = self._tail(stderr_path)
                last_error = RuntimeLaunchError(
                    f"Runtime for role='{process.role}' exited during startup "
                    f"with code {return_code}. See {stderr_path}."
                )
                self._publish_diagnostics(
                    process,
                    profile,
                    precheck,
                    status="failed",
                    launch_args=command,
                    failure_reason=stderr_tail.strip() or str(last_error),
                )
                if attempt == 0 and self._is_address_in_use_error(stderr_tail):
                    self._reassign_port(process, requested_port)
                    continue
                if self._is_memory_error(stderr_tail) and profile_index < len(profiles):
                    self._emit(
                        f"[MODEL_FALLBACK] {process.role} profile={profile.reason} "
                        f"reason=runtime memory initialization failed"
                    )
                    break
                self._emit(f"[MODEL_FAILED] {process.role} {stderr_tail[-500:]}")
                raise last_error

        raise RuntimeLaunchError(
            f"Failed to launch runtime for role='{process.role}': {last_error}"
        )

    def _reassign_port(self, process: RuntimeProcess, requested_port: int) -> None:
        process.port = self._find_available_port(process.port + 1)
        process.metadata = {
            **process.metadata,
            "requested_port": requested_port,
            "port_reassigned": True,
        }

    @staticmethod
    def _port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @classmethod
    def _find_available_port(cls, start: int) -> int:
        for port in range(max(1024, start), 65536):
            if not cls._port_in_use(port):
                return port
        raise RuntimeLaunchError("No available runtime port found.")

    def _build_command(
        self,
        process: RuntimeProcess,
        profile: RuntimeLaunchProfile | None = None,
    ) -> list[str]:
        """Build vLLM OpenAI-compatible server launch command."""
        profile = profile or RuntimeLaunchProfile(
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=self.max_num_seqs,
        )
        command = [
            "python", "-m",
            "vllm.entrypoints.openai.api_server",
            "--model", os.path.expanduser(profile.model_path or process.model_path),
            "--served-model-name", process.model_name,
            "--port", str(process.port),
            "--max-model-len", str(profile.max_model_len),
            "--gpu-memory-utilization", profile.gpu_memory_utilization,
            "--max-num-seqs", str(profile.max_num_seqs),
        ]
        if profile.enforce_eager:
            command.append("--enforce-eager")
        command.extend(self.extra_args)
        return command

    def _launch_profiles(self, process: RuntimeProcess) -> list[RuntimeLaunchProfile]:
        base_gpu = self._parse_gpu_utilization(self.gpu_memory_utilization)
        profiles = [
            RuntimeLaunchProfile(
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_num_seqs=self.max_num_seqs,
                reason="configured",
            ),
            RuntimeLaunchProfile(
                max_model_len=min(self.max_model_len, 4096),
                gpu_memory_utilization=f"{max(base_gpu, 0.95):.2f}",
                max_num_seqs=min(self.max_num_seqs, 16),
                enforce_eager=True,
                reason="reduced_context_cache_recovery",
            ),
            RuntimeLaunchProfile(
                max_model_len=min(self.max_model_len, 2048),
                gpu_memory_utilization=f"{min(base_gpu, 0.85):.2f}",
                max_num_seqs=min(self.max_num_seqs, 4),
                enforce_eager=True,
                reason="low_pressure",
            ),
        ]
        quantized = process.metadata.get("quantized_model_path")
        if isinstance(quantized, str) and quantized.strip():
            profiles.append(
                RuntimeLaunchProfile(
                    max_model_len=min(self.max_model_len, 4096),
                    gpu_memory_utilization=f"{min(base_gpu, 0.85):.2f}",
                    max_num_seqs=min(self.max_num_seqs, 8),
                    enforce_eager=True,
                    model_path=quantized,
                    reason="quantized_variant",
                )
            )
        return profiles

    def _runtime_precheck(
        self,
        process: RuntimeProcess,
        profile: RuntimeLaunchProfile,
        profile_index: int,
        profile_count: int,
    ) -> dict[str, object]:
        memory = self._gpu_memory_snapshot()
        model_size_mb = self._estimate_model_size_mb(profile.model_path or process.model_path)
        kv_cache_mb = self._estimate_kv_cache_mb(profile)
        overhead_mb = 1536
        required_mb = (
            None
            if model_size_mb is None
            else int(model_size_mb * 1.08 + kv_cache_mb + overhead_mb)
        )
        requested_allocation_mb = (
            int(memory.total_mb * self._parse_gpu_utilization(profile.gpu_memory_utilization))
            if memory.total_mb is not None
            else None
        )
        usable_vram_mb = memory.free_mb
        if requested_allocation_mb is not None and memory.free_mb is not None:
            usable_vram_mb = min(memory.free_mb, requested_allocation_mb)
        fits = True
        failure_reason: str | None = None
        if (
            requested_allocation_mb is not None
            and memory.free_mb is not None
            and requested_allocation_mb > memory.free_mb
        ):
            fits = False
            failure_reason = (
                f"gpu_memory_utilization requests {requested_allocation_mb}MB "
                f"but only {memory.free_mb}MB is free"
            )
        elif usable_vram_mb is not None and required_mb is not None and required_mb > usable_vram_mb:
            fits = False
            failure_reason = (
                f"insufficient VRAM for {process.role}: "
                f"required={required_mb}MB usable={usable_vram_mb}MB"
            )
        precheck = {
            "fits": fits,
            "failure_reason": failure_reason,
            "free_vram_mb": memory.free_mb,
            "used_vram_mb": memory.used_mb,
            "total_vram_mb": memory.total_mb,
            "memory_source": memory.source,
            "memory_error": memory.error,
            "model_size_mb": model_size_mb,
            "kv_cache_requirement_mb": kv_cache_mb,
            "required_vram_mb": required_mb,
            "requested_allocation_mb": requested_allocation_mb,
            "usable_vram_mb": usable_vram_mb,
            "profile_index": profile_index,
            "profile_count": profile_count,
        }
        self._publish_diagnostics(
            process,
            profile,
            precheck,
            status="precheck",
            launch_args=[],
            failure_reason=failure_reason,
        )
        self._emit(
            f"[RUNTIME_PRECHECK] {process.role} profile={profile.reason} "
            f"fits={fits}"
        )
        self._emit(
            f"[VRAM_AVAILABLE] free={memory.free_mb}MB used={memory.used_mb}MB "
            f"total={memory.total_mb}MB"
        )
        self._emit(
            f"[VRAM_REQUIRED] required={required_mb}MB model={model_size_mb}MB "
            f"kv_cache={kv_cache_mb}MB"
        )
        return precheck

    def _publish_diagnostics(
        self,
        process: RuntimeProcess,
        profile: RuntimeLaunchProfile,
        precheck: dict[str, object],
        *,
        status: str,
        launch_args: list[str],
        failure_reason: str | None,
    ) -> None:
        diagnostics: dict[str, object] = {
            "role": process.role,
            "target_model": process.model_name,
            "model_path": os.path.expanduser(profile.model_path or process.model_path),
            "free_vram": precheck.get("free_vram_mb"),
            "used_vram": precheck.get("used_vram_mb"),
            "total_vram": precheck.get("total_vram_mb"),
            "model_size": precheck.get("model_size_mb"),
            "kv_cache_requirement": precheck.get("kv_cache_requirement_mb"),
            "vram_required": precheck.get("required_vram_mb"),
            "vram_requested_allocation": precheck.get("requested_allocation_mb"),
            "vram_usable": precheck.get("usable_vram_mb"),
            "launch_args": launch_args,
            "load_status": status,
            "fallback_status": profile.reason,
            "failure_reason": failure_reason,
            "profile": asdict(profile),
        }
        self.last_diagnostics = diagnostics
        process.metadata = {
            **process.metadata,
            "runtime_diagnostics": diagnostics,
        }
        if self.diagnostics_callback:
            self.diagnostics_callback(diagnostics)

    def _gpu_memory_snapshot(self) -> RuntimeMemorySnapshot:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return RuntimeMemorySnapshot(None, None, None, error=str(exc))
        if result.returncode != 0 or not result.stdout.strip():
            return RuntimeMemorySnapshot(
                None,
                None,
                None,
                source="nvidia-smi",
                error=result.stderr.strip() or "nvidia-smi returned no data",
            )
        try:
            total, used, free = [
                int(part.strip())
                for part in result.stdout.splitlines()[0].split(",")[:3]
            ]
        except (ValueError, IndexError) as exc:
            return RuntimeMemorySnapshot(
                None,
                None,
                None,
                source="nvidia-smi",
                error=f"failed to parse nvidia-smi output: {exc}",
            )
        return RuntimeMemorySnapshot(
            total_mb=total,
            used_mb=used,
            free_mb=free,
            source="nvidia-smi",
        )

    @staticmethod
    def _estimate_model_size_mb(model_path: str) -> int | None:
        root = Path(os.path.expanduser(model_path))
        if not root.exists():
            return None
        if root.is_file():
            return max(1, round(root.stat().st_size / (1024 * 1024)))
        suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
        total = 0
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                total += path.stat().st_size
        if total <= 0:
            return None
        return max(1, round(total / (1024 * 1024)))

    @staticmethod
    def _estimate_kv_cache_mb(profile: RuntimeLaunchProfile) -> int:
        token_factor = (
            (profile.max_model_len / RuntimeLauncher.DEFAULT_MAX_MODEL_LEN)
            * (profile.max_num_seqs / RuntimeLauncher.DEFAULT_MAX_NUM_SEQS)
        )
        return max(512, round(6144 * token_factor))

    @staticmethod
    def _parse_gpu_utilization(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError:
            return 0.90
        return min(1.0, max(0.05, parsed))

    def _wait_for_early_failure(self, proc: subprocess.Popen) -> int | None:
        return_code = proc.poll()
        if return_code is not None:
            return return_code
        deadline = time.monotonic() + self.startup_failure_window
        while time.monotonic() < deadline:
            return_code = proc.poll()
            if return_code is not None:
                return return_code
            time.sleep(0.2)
        return None

    @staticmethod
    def _is_address_in_use_error(text: str) -> bool:
        return (
            "Address already in use" in text
            or "Errno 98" in text
            or "EADDRINUSE" in text
        )

    @staticmethod
    def _is_memory_error(text: str) -> bool:
        lowered = text.lower()
        return (
            "no available memory for the cache blocks" in lowered
            or "out of memory" in lowered
            or "cuda out of memory" in lowered
        )

    def _emit(self, message: str) -> None:
        self.telemetry(message)

    def _log_paths(self, process: RuntimeProcess) -> tuple[str, str]:
        """Return (stdout_path, stderr_path) for this process role."""
        role = process.role.lower()
        return (
            os.path.join(self.log_dir, f"{role}_stdout.log"),
            os.path.join(self.log_dir, f"{role}_stderr.log"),
        )

    @staticmethod
    def _tail(path: str, max_chars: int = 4000) -> str:
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_chars), os.SEEK_SET)
                return handle.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
