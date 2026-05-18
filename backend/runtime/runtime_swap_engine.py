from __future__ import annotations
import time
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
    ) -> None:
        self.launcher = launcher
        self.shutdown_manager = shutdown
        self.swap_delay_seconds = swap_delay_seconds
        self.active_process: RuntimeProcess | None = None
        self.swap_history: list[RuntimeProcess] = []

    def swap(self, next_process: RuntimeProcess) -> RuntimeProcess:
        """
        Shutdown current runtime, wait for VRAM cleanup,
        then launch next runtime.
        """
        if self.active_process is not None:
            try:
                self.shutdown_manager.shutdown(self.active_process)
                self.swap_history.append(self.active_process)
                time.sleep(self.swap_delay_seconds)
            except Exception as exc:
                raise RuntimeSwapError(
                    f"Failed to shutdown runtime for "
                    f"role={self.active_process.role}: {exc}"
                ) from exc

        try:
            launched = self.launcher.launch(next_process)
        except Exception as exc:
            raise RuntimeSwapError(
                f"Failed to launch runtime for "
                f"role={next_process.role}: {exc}"
            ) from exc

        self.active_process = launched
        return launched

    def shutdown_active(self) -> None:
        """Shutdown active runtime without replacement."""
        if self.active_process is None:
            return

        try:
            self.shutdown_manager.shutdown(self.active_process)
            self.swap_history.append(self.active_process)
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