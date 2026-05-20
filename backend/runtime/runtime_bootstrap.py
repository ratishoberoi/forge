from __future__ import annotations
from backend.runtime.runtime_health import RuntimeHealth
from backend.runtime.runtime_launcher import RuntimeLauncher
from backend.runtime.runtime_process import RuntimeProcess


class RuntimeBootstrapError(Exception):
    """Raised when a runtime fails to boot within the allowed timeout."""


class RuntimeBootstrap:
    """
    Launches a runtime process and waits for it to become ready.
    Responsibilities:
    - delegate launch to RuntimeLauncher
    - poll readiness via RuntimeHealth
    - raise RuntimeBootstrapError if timeout exceeded
    - support optional pre-boot health check
    """

    DEFAULT_TIMEOUT = 300

    def __init__(
        self,
        *,
        launcher: RuntimeLauncher,
        health: RuntimeHealth,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.launcher = launcher
        self.health = health
        self.timeout = timeout

    def boot(self, process: RuntimeProcess) -> RuntimeProcess:
        """
        Launch process and wait until ready.
        Raises RuntimeBootstrapError if runtime does not become ready.
        """
        if process.is_running:
            raise RuntimeBootstrapError(
                f"Process for role '{process.role}' is already running "
                f"(pid={process.pid})."
            )

        if self.health.is_ready(process.port):
            raise RuntimeBootstrapError(
                f"Port {process.port} is already in use before launch. "
                f"Another process may be running."
            )

        launched = self.launcher.launch(process)

        ready = self.health.wait_until_ready(
            port=process.port,
            model_name=process.model_name,
            timeout=self.timeout,
        )

        if not ready:
            raise RuntimeBootstrapError(
                f"Runtime '{process.role}' failed to become ready "
                f"on port {process.port} within {self.timeout}s."
            )

        return launched

    def boot_and_verify(self, process: RuntimeProcess) -> RuntimeProcess:
        """
        Boot then do a final single ping to confirm readiness.
        Extra safety check for flaky runtimes.
        """
        launched = self.boot(process)

        if not self.health.is_ready(process.port, model_name=process.model_name):
            raise RuntimeBootstrapError(
                f"Runtime '{process.role}' passed wait_until_ready "
                f"but failed final verification ping on port {process.port}."
            )

        return launched
