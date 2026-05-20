from __future__ import annotations
import time
import requests


class RuntimeHealthError(Exception):
    """Raised when health checks fail unexpectedly."""


class RuntimeHealth:
    """
    Runtime readiness checker for OpenAI-compatible vLLM servers.
    Uses /v1/models because it is significantly more reliable
    than /health across vLLM versions.
    Responsibilities:
    - poll until runtime is ready to serve requests
    - confirm runtime has stopped after shutdown
    - provide single non-blocking readiness check
    """

    DEFAULT_TIMEOUT = 300
    DEFAULT_POLL_INTERVAL = 2
    MODELS_PATH = "/v1/models"

    def __init__(
        self,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self.poll_interval = poll_interval

    def wait_until_ready(
        self,
        *,
        port: int,
        model_name: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> bool:
        """
        Poll /v1/models until a valid model registry is returned or timeout.
        Returns True if ready, False if timed out.
        """
        url = self._models_url(port)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self._ping(url, model_name=model_name):
                return True
            time.sleep(self.poll_interval)

        return False

    def is_ready(self, port: int, model_name: str | None = None) -> bool:
        """
        Single non-blocking check.
        Returns True if /v1/models returns a valid registry right now.
        """
        return self._ping(self._models_url(port), model_name=model_name)

    def wait_until_stopped(
        self,
        *,
        port: int,
        timeout: int = 30,
    ) -> bool:
        """
        Poll until /v1/models stops responding.
        Useful after SIGTERM to confirm clean shutdown.
        Returns True if stopped within timeout.
        """
        url = self._models_url(port)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if not self._ping(url):
                return True
            time.sleep(self.poll_interval)

        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _models_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}{self.MODELS_PATH}"

    def _ping(self, url: str, model_name: str | None = None) -> bool:
        try:
            response = requests.get(
                url,
                timeout=5,
            )

            if response.status_code != 200:
                return False

            data = response.json()

            return self._valid_model_registry(data, model_name=model_name)
        except (ValueError, requests.exceptions.RequestException):
            return False

    @staticmethod
    def _valid_model_registry(data: object, model_name: str | None = None) -> bool:
        if not isinstance(data, dict):
            return False

        models = data.get("data")
        if not isinstance(models, list) or not models:
            return False

        ids: list[str] = []
        for model in models:
            if not isinstance(model, dict):
                return False
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                return False
            ids.append(model_id)

        return model_name is None or model_name in ids
