from __future__ import annotations
import os
import signal
import time
from backend.runtime.runtime_process import RuntimeProcess


class RuntimeShutdownError(Exception):
    """Raised when runtime shutdown fails."""


class RuntimeShutdown:
    """
    Gracefully terminates heavyweight inference runtimes.
    Designed for sequential one-GPU orchestration.
    Responsibilities:
    - send SIGTERM and wait for clean exit
    - escalate to SIGKILL if process still alive
    - handle already-dead processes gracefully
    - mark process stopped after shutdown
    """

    DEFAULT_SHUTDOWN_WAIT = 8.0

    def __init__(
        self,
        *,
        shutdown_wait: float = DEFAULT_SHUTDOWN_WAIT,
        escalate_to_sigkill: bool = True,
    ) -> None:
        self.shutdown_wait = shutdown_wait
        self.escalate_to_sigkill = escalate_to_sigkill

    def shutdown(self, process: RuntimeProcess) -> None:
        """
        Terminate process via SIGTERM.
        Waits shutdown_wait seconds then escalates to SIGKILL if still alive.
        No-op if process has no PID or is already inactive.
        """
        if process.pid is None:
            return

        if not process.active:
            return

        # Send SIGTERM
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # Already dead — just mark stopped
            process.mark_stopped()
            return
        except PermissionError as exc:
            raise RuntimeShutdownError(
                f"Permission denied terminating pid={process.pid} "
                f"for role='{process.role}'."
            ) from exc

        # Wait for clean exit
        time.sleep(self.shutdown_wait)

        # Escalate to SIGKILL if still alive
        if self.escalate_to_sigkill:
            self._sigkill_if_alive(process)

        process.mark_stopped()

    def shutdown_nowait(self, process: RuntimeProcess) -> None:
        """
        Send SIGTERM without waiting.
        Useful when caller manages its own wait/poll loop.
        """
        if process.pid is None or not process.active:
            return

        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeShutdownError(
                f"Permission denied terminating pid={process.pid} "
                f"for role='{process.role}'."
            ) from exc
        finally:
            process.mark_stopped()

    def is_alive(self, process: RuntimeProcess) -> bool:
        """
        Check if process is still alive via kill(pid, 0).
        Returns False if process has no PID or is not found.
        """
        if process.pid is None:
            return False
        try:
            os.kill(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we can't signal it
            return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sigkill_if_alive(self, process: RuntimeProcess) -> None:
        """Send SIGKILL if process is still responding to signal 0."""
        try:
            os.kill(process.pid, 0)
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass