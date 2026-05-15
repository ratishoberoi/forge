from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeEndpoint:
    role: str
    model_name: str
    base_url: str
    api_path: str = "/v1/chat/completions"
    timeout_seconds: float = 60.0
    max_retries: int = 3

    @property
    def full_url(self) -> str:
        """Convenience: base_url + api_path."""
        return f"{self.base_url.rstrip('/')}{self.api_path}"

    def with_base_url(self, base_url: str) -> RuntimeEndpoint:
        """Return a copy with a different base_url — useful for testing."""
        return RuntimeEndpoint(
            role=self.role,
            model_name=self.model_name,
            base_url=base_url,
            api_path=self.api_path,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )


RUNTIME_REGISTRY: dict[str, RuntimeEndpoint] = {
    "PRIMARY_CODER": RuntimeEndpoint(
        role="PRIMARY_CODER",
        model_name="qwen-primary",
        base_url="http://127.0.0.1:8000",
    ),
    "JUDGE": RuntimeEndpoint(
        role="JUDGE",
        model_name="qwen-judge",
        base_url="http://127.0.0.1:8001",
    ),
    "EMBEDDER": RuntimeEndpoint(
        role="EMBEDDER",
        model_name="bge-embedder",
        base_url="http://127.0.0.1:8002",
        api_path="/v1/embeddings",
    ),
}


def get_endpoint(role: str) -> RuntimeEndpoint:
    """
    Fetch endpoint by role. Raises KeyError with a clear message if missing.
    """
    try:
        return RUNTIME_REGISTRY[role.upper()]
    except KeyError:
        available = list(RUNTIME_REGISTRY.keys())
        raise KeyError(
            f"Unknown runtime role '{role}'. Available roles: {available}"
        ) from None


def register_endpoint(endpoint: RuntimeEndpoint) -> None:
    """
    Dynamically register a new endpoint into the registry.
    Overwrites if role already exists.
    """
    RUNTIME_REGISTRY[endpoint.role.upper()] = endpoint