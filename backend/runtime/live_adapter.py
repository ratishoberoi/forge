from __future__ import annotations
from backend.runtime.runtime_client import RuntimeClient, RuntimeClientError
from backend.runtime.runtime_registry import get_endpoint, RuntimeEndpoint


class LiveAdapterError(Exception):
    """Raised when LiveCognitionAdapter encounters a non-recoverable error."""


class LiveCognitionAdapter:
    """
    High-level adapter between cognition roles and runtime endpoints.
    Responsibilities:
    - resolve role to RuntimeEndpoint via registry
    - build OpenAI-compatible message payloads
    - delegate HTTP to RuntimeClient
    - surface clean errors to callers
    """

    def __init__(self, client: RuntimeClient | None = None) -> None:
        self.client = client or RuntimeClient()

    async def execute(
        self,
        *,
        role: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """
        Execute a prompt against the endpoint registered for the given role.
        Optionally prepend a system message.
        """
        endpoint = self._resolve_role(role)
        messages = self._build_messages(prompt, system_prompt)

        try:
            return await self.client.chat_completion(
                base_url=endpoint.base_url,
                model=endpoint.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RuntimeClientError as exc:
            raise LiveAdapterError(
                f"Execution failed for role '{role}' "
                f"at {endpoint.base_url}: {exc}"
            ) from exc

    async def execute_with_endpoint(
        self,
        *,
        endpoint: RuntimeEndpoint,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """
        Execute directly against a given endpoint — bypasses registry lookup.
        Useful for testing or one-off overrides.
        """
        messages = self._build_messages(prompt, system_prompt)

        try:
            return await self.client.chat_completion(
                base_url=endpoint.base_url,
                model=endpoint.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RuntimeClientError as exc:
            raise LiveAdapterError(
                f"Execution failed for endpoint '{endpoint.role}' "
                f"at {endpoint.base_url}: {exc}"
            ) from exc

    async def health_check(self, role: str) -> bool:
        """
        Check if the endpoint for the given role is reachable.
        Non-raising — returns False on any failure.
        """
        try:
            endpoint = self._resolve_role(role)
            return await self.client.health_check(endpoint.base_url)
        except LiveAdapterError:
            return False

    def _resolve_role(self, role: str) -> RuntimeEndpoint:
        """Resolve role string to RuntimeEndpoint. Raises LiveAdapterError if missing."""
        try:
            return get_endpoint(role)
        except KeyError as exc:
            raise LiveAdapterError(str(exc)) from exc

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: str | None,
    ) -> list[dict]:
        """Build OpenAI-compatible messages list."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages