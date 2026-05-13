from __future__ import annotations
from dataclasses import dataclass
from backend.llm.providers.openai_compatible import OpenAICompatibleClient
from backend.llm.router import ModelRole


@dataclass(slots=True)
class LiveCognitionResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


class LiveCognition:
    """
    External API-backed cognition runtime.
    Uses:
    - vLLM OpenAI server
    - LM Studio
    - future distributed cognition endpoints
    """

    def __init__(
        self,
        primary_client: OpenAICompatibleClient,
        judge_client: OpenAICompatibleClient | None = None,
    ) -> None:
        self.primary_client = primary_client
        self.judge_client = judge_client or primary_client

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_role: ModelRole = ModelRole.PRIMARY_CODER,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        agent_id: str | None = None,
    ) -> LiveCognitionResponse:
        client = (
            self.judge_client
            if model_role == ModelRole.JUDGE
            else self.primary_client
        )
        content = await client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LiveCognitionResponse(
            content=content,
            model=model_role.value,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="stop",
        )
    async def critique(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
    ) -> LiveCognitionResponse:
        """Dispatch directly to judge client at temperature=0.0."""
        content = await self.judge_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return LiveCognitionResponse(content=content, model=ModelRole.JUDGE.value)

    async def __aenter__(self) -> LiveCognition:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self.primary_client.close()
        if self.judge_client is not self.primary_client:
            await self.judge_client.close()