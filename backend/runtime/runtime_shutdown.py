from __future__ import annotations
import os
import signal
from backend.runtime.runtime_process import RuntimeProcess


class RuntimeShutdownError(Exception):
    """Raised when a runtime process cannot be shut down cleanly."""


class RuntimeShutdown:
    """
    Terminates runtime inference processes.
    Responsibilities:
    - send SIGTERM to process
    - optionally escalate to SIGKILL
    - update RuntimeProcess state
    - handle already-dead processes gracefully
    """

    def __init__(self, escalate_to_sigkill: bool = False) -> None:
        self.escalate_to_sigkill = escalate_to_sigkill

    def shutdown(self, process: RuntimeProcess) -> None:
        """
        Send SIGTERM to process. Marks process inactive.
        No-op if process has no PID.
        """
        if process.pid is None:
            return

        if not process.active:
            return

        self._terminate(process)
        process.mark_stopped()

    def _terminate(self, process: RuntimeProcess) -> None:
        """Send SIGTERM, optionally escalate to SIGKILL."""
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process already dead — mark stopped and move on
            process.mark_stopped()
            return
        except PermissionError as exc:
            raise RuntimeShutdownError(
                f"Permission denied terminating pid={process.pid} "
                f"for role '{process.role}'."
            ) from exc

        if self.escalate_to_sigkill:
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass