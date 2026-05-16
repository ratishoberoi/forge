from __future__ import annotations
from backend.runtime.runtime_health import RuntimeHealth
from backend.runtime.runtime_process import RuntimeProcess
from backend.runtime.runtime_shutdown import RuntimeShutdown


class RuntimeTeardownError(Exception):
    """Raised when teardown cannot complete cleanly."""


class RuntimeTeardown:
    """
    Gracefully terminates a runtime process.
    Responsibilities:
    - send shutdown signal via RuntimeShutdown
    - optionally wait until port stops responding
    - handle already-stopped processes gracefully
    - report teardown outcome
    """

    DEFAULT_STOP_TIMEOUT = 30

    def __init__(
        self,
        *,
        shutdown: RuntimeShutdown,
        health: RuntimeHealth | None = None,
        stop_timeout: int = DEFAULT_STOP_TIMEOUT,
    ) -> None:
        self.shutdown = shutdown
        self.health = health
        self.stop_timeout = stop_timeout

    def teardown(self, process: RuntimeProcess) -> None:
        """
        Shutdown process. No-op if already inactive.
        Optionally waits for port to stop responding.
        """
        if not process.active:
            return

        self.shutdown.shutdown(process)

        if self.health is not None:
            stopped = self.health.wait_until_stopped(
                port=process.port,
                timeout=self.stop_timeout,
            )
            if not stopped:
                raise RuntimeTeardownError(
                    f"Runtime '{process.role}' on port {process.port} "
                    f"did not stop within {self.stop_timeout}s after SIGTERM."
                )

    def teardown_all(self, processes: list[RuntimeProcess]) -> dict[str, bool]:
        """
        Teardown multiple processes.
        Returns dict of role → success (True) / failure (False).
        """
        results: dict[str, bool] = {}
        for process in processes:
            try:
                self.teardown(process)
                results[process.role] = True
            except (RuntimeTeardownError, Exception):
                results[process.role] = False
        return results