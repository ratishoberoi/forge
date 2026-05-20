from __future__ import annotations
import time
from collections.abc import Callable
from backend.runtime.runtime_launcher import RuntimeLauncher
from backend.runtime.runtime_process import RuntimeProcess
from backend.runtime.runtime_shutdown import RuntimeShutdown


class RuntimeSwapError(Exception):
    """Raised when a runtime swap cannot be completed."""


class RuntimeSwapEngine:
    """
    Real sequential heavyweight runtime orchestration.
    Guarantees:
    - only one heavyweight runtime active at a time
    - previous runtime shutdown before next launch
    - VRAM cleanup wait between swaps
    - swap history preserved
    """

    def __init__(
        self,
        launcher: RuntimeLauncher,
        shutdown: RuntimeShutdown,
        swap_delay_seconds: float = 20.0,
        max_launch_attempts: int = 2,
        telemetry: Callable[[str], None] | None = None,
    ) -> None:
        self.launcher = launcher
        self.shutdown_manager = shutdown
        self.swap_delay_seconds = swap_delay_seconds
        self.max_launch_attempts = max(1, max_launch_attempts)
        self.telemetry = telemetry or print
        self.active_process: RuntimeProcess | None = None
        self.swap_history: list[RuntimeProcess] = []

    def swap(self, next_process: RuntimeProcess) -> RuntimeProcess:
        """
        Shutdown current runtime, wait for VRAM cleanup,
        then launch next runtime.
        """
        if self.active_process is not None:
            try:
                self._emit(
                    f"[SWAP] shutting down {self.active_process.role} "
                    f"({self.active_process.model_name})"
                )
                self.shutdown_manager.shutdown(self.active_process)
                self.swap_history.append(self.active_process)
                self._emit(f"[SWAP] shutdown complete {self.active_process.role}")
                time.sleep(self.swap_delay_seconds)
                self.active_process = None
            except Exception as exc:
                raise RuntimeSwapError(
                    f"Failed to shutdown runtime for "
                    f"role={self.active_process.role}: {exc}"
                ) from exc

        launched = self._launch_with_recovery(next_process)

        self.active_process = launched
        return launched

    def shutdown_active(self) -> None:
        """Shutdown active runtime without replacement."""
        if self.active_process is None:
            return

        try:
            self._emit(
                f"[SWAP] shutting down {self.active_process.role} "
                f"({self.active_process.model_name})"
            )
            self.shutdown_manager.shutdown(self.active_process)
            self.swap_history.append(self.active_process)
            self._emit(f"[SWAP] shutdown complete {self.active_process.role}")
            time.sleep(self.swap_delay_seconds)
        finally:
            self.active_process = None

    @property
    def has_active(self) -> bool:
        return self.active_process is not None

    @property
    def swap_count(self) -> int:
        return len(self.swap_history)

    @property
    def active_role(self) -> str | None:
        return self.active_process.role if self.active_process else None

    def history_roles(self) -> list[str]:
        return [p.role for p in self.swap_history]

    def _launch_with_recovery(self, process: RuntimeProcess) -> RuntimeProcess:
        last_error: Exception | None = None
        for attempt in range(1, self.max_launch_attempts + 1):
            try:
                self._emit(
                    f"[SWAP] launching {process.role} "
                    f"({process.model_name}) attempt {attempt}/"
                    f"{self.max_launch_attempts}"
                )
                launched = self.launcher.launch(process)
                self._emit(
                    f"[SWAP] launched {process.role} "
                    f"pid={launched.pid} pgid={launched.pgid}"
                )
                return launched
            except Exception as exc:
                last_error = exc
                self._emit(
                    f"[SWAP] launch failed {process.role} "
                    f"attempt {attempt}/{self.max_launch_attempts}: {exc}"
                )
                self._cleanup_failed_launch(process)
                if attempt < self.max_launch_attempts:
                    time.sleep(self.swap_delay_seconds)

        raise RuntimeSwapError(
            f"Failed to launch runtime for role={process.role}: {last_error}"
        )

    def _cleanup_failed_launch(self, process: RuntimeProcess) -> None:
        if process.pid is None:
            return
        try:
            self.shutdown_manager.shutdown_nowait(process)
        except AttributeError:
            self.shutdown_manager.shutdown(process)
        except Exception as exc:
            self._emit(f"[SWAP] stale runtime cleanup failed {process.role}: {exc}")

    def _emit(self, message: str) -> None:
        self.telemetry(message)
