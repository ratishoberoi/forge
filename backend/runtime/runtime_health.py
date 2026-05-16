from __future__ import annotations
import time
import requests


class RuntimeHealthError(Exception):
    """Raised when health check encounters an unexpected error."""


class RuntimeHealth:
    """
    Polls runtime inference endpoint for readiness.
    Responsibilities:
    - wait until /health returns 200
    - respect timeout
    - support configurable poll interval
    - provide instant check without waiting
    """

    DEFAULT_TIMEOUT = 300
    DEFAULT_POLL_INTERVAL = 2
    HEALTH_PATH = "/health"

    def __init__(
        self,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self.poll_interval = poll_interval

    def wait_until_ready(
        self,
        *,
        port: int,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> bool:
        """
        Poll /health until 200 or timeout.
        Returns True if ready, False if timed out.
        """
        url = self._health_url(port)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self._ping(url):
                return True
            time.sleep(self.poll_interval)

        return False

    def is_ready(self, port: int) -> bool:
        """
        Single non-blocking check.
        Returns True if /health returns 200 right now.
        """
        return self._ping(self._health_url(port))

    def wait_until_stopped(
        self,
        *,
        port: int,
        timeout: int = 30,
    ) -> bool:
        """
        Poll until /health stops responding.
        Useful after SIGTERM to confirm clean shutdown.
        Returns True if stopped within timeout.
        """
        url = self._health_url(port)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if not self._ping(url):
                return True
            time.sleep(self.poll_interval)

        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _health_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}{self.HEALTH_PATH}"

    def _ping(self, url: str) -> bool:
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False