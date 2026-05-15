from __future__ import annotations
import httpx


class RuntimeClientError(Exception):
    """Raised when RuntimeClient encounters a non-recoverable error."""


class RuntimeClient:
    """
    Low-level async HTTP client for OpenAI-compatible inference endpoints.
    Responsibilities:
    - send chat completion requests
    - handle HTTP errors cleanly
    - validate response structure
    - support configurable timeouts and retries
    """

    DEFAULT_TIMEOUT = 120.0

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    async def chat_completion(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        extra_body: dict | None = None,
    ) -> str:
        """
        Send a chat completion request.
        Returns the assistant message content as a string.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{base_url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeClientError(
                    f"HTTP {exc.response.status_code} from {base_url}: "
                    f"{exc.response.text[:200]}"
                ) from exc
            except httpx.RequestError as exc:
                raise RuntimeClientError(
                    f"Request failed to {base_url}: {exc}"
                ) from exc

        return self._extract_content(response.json(), base_url)

    def _extract_content(self, payload: dict, base_url: str) -> str:
        """Safely extract assistant content from OpenAI-compatible response."""
        try:
            choices = payload["choices"]
            if not choices:
                raise RuntimeClientError(
                    f"Empty choices in response from {base_url}."
                )
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise RuntimeClientError(
                    f"Unexpected content type {type(content)} from {base_url}."
                )
            return content
        except (KeyError, IndexError) as exc:
            raise RuntimeClientError(
                f"Malformed response structure from {base_url}: {exc}"
            ) from exc

    async def health_check(self, base_url: str) -> bool:
        """
        Ping /health endpoint. Returns True if server is reachable.
        Non-raising — safe to call before inference.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{base_url.rstrip('/')}/health")
                return response.status_code == 200
        except httpx.RequestError:
            return False