from __future__ import annotations
import httpx


class OpenAICompatibleClient:
    """
    OpenAI-compatible local inference client.
    Supports:
    - vLLM
    - LM Studio
    - Ollama OpenAI mode
    - future distributed endpoints

    NOTE: "OpenAI-compatible" refers only to the HTTP API format
    (/v1/chat/completions). No data leaves your machine.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"Empty choices in response from {self.base_url}. "
                f"Raw response: {data}"
            )

        return choices[0]["message"]["content"]

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()