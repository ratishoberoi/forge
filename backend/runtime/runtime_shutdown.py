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
    Supports:
    - process-group termination
    - graceful SIGTERM
    - forced SIGKILL escalation
    """

    DEFAULT_SHUTDOWN_WAIT = 2.0

    def __init__(
        self,
        *,
        shutdown_wait: float = DEFAULT_SHUTDOWN_WAIT,
        escalate_to_sigkill: bool = True,
    ) -> None:
        self.shutdown_wait = shutdown_wait
        self.escalate_to_sigkill = escalate_to_sigkill

    def shutdown(self, process: RuntimeProcess) -> None:
        if process.pid is None:
            return
        if not process.active:
            return

        try:
            self._terminate_group(process, signal.SIGTERM)
        except ProcessLookupError:
            process.mark_stopped()
            time.sleep(3)
            return
        except PermissionError as exc:
            raise RuntimeShutdownError(
                f"Permission denied terminating pid={process.pid}"
            ) from exc

        time.sleep(self.shutdown_wait)

        if self.escalate_to_sigkill and self.is_alive(process):
            try:
                self._terminate_group(process, signal.SIGKILL)
            except ProcessLookupError:
                pass

        process.mark_stopped()
        time.sleep(3)

    def shutdown_nowait(self, process: RuntimeProcess) -> None:
        if process.pid is None or not process.active:
            return

        try:
            self._terminate_group(process, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeShutdownError(
                f"Permission denied terminating pid={process.pid}"
            ) from exc
        finally:
            process.mark_stopped()
            time.sleep(3)

    def is_alive(self, process: RuntimeProcess) -> bool:
        if process.pid is None:
            return False
        try:
            if process.pgid is not None:
                os.killpg(process.pgid, 0)
            else:
                os.kill(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _terminate_group(
        self,
        process: RuntimeProcess,
        sig: signal.Signals,
    ) -> None:
        """
        Kill entire vLLM process group.
        This is REQUIRED because vLLM spawns child workers.
        """
        if process.pgid is not None:
            os.killpg(process.pgid, sig)
        else:
            os.kill(process.pid, sig)