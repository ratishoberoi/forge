from __future__ import annotations
from backend.runtime.runtime_launcher import RuntimeLauncher
from backend.runtime.runtime_process import RuntimeProcess
from backend.runtime.runtime_shutdown import RuntimeShutdown


class RuntimeSwapError(Exception):
    """Raised when a runtime swap cannot be completed."""


class RuntimeSwapEngine:
    """
    Orchestrates sequential runtime lifecycle swaps.
    Pattern:
        spawn runtime
        → execute cognition
        → persist artifact
        → shutdown runtime
        → spawn next runtime

    Responsibilities:
    - track active process
    - shutdown current before launching next
    - maintain swap history
    - enforce single-active invariant
    """

    def __init__(
        self,
        launcher: RuntimeLauncher,
        shutdown: RuntimeShutdown,
    ) -> None:
        self.launcher = launcher
        self.shutdown_manager = shutdown
        self.active_process: RuntimeProcess | None = None
        self.swap_history: list[RuntimeProcess] = []

    def swap(self, next_process: RuntimeProcess) -> RuntimeProcess:
        """
        Shutdown current process (if any) and launch next.
        Returns the newly launched RuntimeProcess.
        """
        if self.active_process is not None:
            self.shutdown_manager.shutdown(self.active_process)
            self.swap_history.append(self.active_process)

        launched = self.launcher.launch(next_process)
        self.active_process = launched
        return launched

    def shutdown_active(self) -> None:
        """
        Shutdown the currently active process without launching a replacement.
        No-op if no active process.
        """
        if self.active_process is None:
            return
        self.shutdown_manager.shutdown(self.active_process)
        self.swap_history.append(self.active_process)
        self.active_process = None

    @property
    def has_active(self) -> bool:
        """True if there is a currently active process."""
        return self.active_process is not None

    @property
    def swap_count(self) -> int:
        """Number of swaps performed so far."""
        return len(self.swap_history)

    @property
    def active_role(self) -> str | None:
        """Role of the currently active process, or None."""
        return self.active_process.role if self.active_process else None

    def history_roles(self) -> list[str]:
        """Ordered list of roles that have been swapped out."""
        return [p.role for p in self.swap_history]