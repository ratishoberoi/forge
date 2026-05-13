from __future__ import annotations
from dataclasses import dataclass
from backend.api.schemas.chat import ChatCompletionRequest, ChatMessage
from backend.llm.router import ModelRole
from backend.llm.service import ChatCompletionService


@dataclass(slots=True)
class CognitionResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None


class CognitionAdapter:
    """
    Agent-facing cognition abstraction.
    Responsibilities:
    - hide OpenAI/chat schema details
    - build structured prompts
    - dispatch to cognition runtime
    - support role-based model routing
    - future retry/judge orchestration
    """

    def __init__(self, service: ChatCompletionService) -> None:
        self.service = service

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_role: ModelRole = ModelRole.PRIMARY_CODER,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        agent_id: str | None = None,
    ) -> CognitionResponse:
        request = ChatCompletionRequest(
            model=model_role.value,
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            agent_id=agent_id,
        )
        response = await self.service.create_chat_completion(request)

        if not response.choices:
            raise RuntimeError(
                f"Empty choices returned by service for role={model_role.value}"
            )

        choice = response.choices[0]
        return CognitionResponse(
            content=choice.message.content,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            finish_reason=choice.finish_reason,
        )

    async def critique(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        agent_id: str | None = None,
    ) -> CognitionResponse:
        return await self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_role=ModelRole.JUDGE,
            temperature=0.0,
            max_tokens=max_tokens,
            agent_id=agent_id,
        )