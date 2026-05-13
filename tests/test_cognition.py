import pytest
from backend.api.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from backend.llm.router import ModelRole
from backend.runtime.cognition import CognitionAdapter


class FakeChatCompletionService:
    async def create_chat_completion(self, request):
        return ChatCompletionResponse(
            id="test",
            object="chat.completion",
            created=0,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="hello"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


class EmptyChoicesService:
    async def create_chat_completion(self, request):
        return ChatCompletionResponse(
            id="test",
            object="chat.completion",
            created=0,
            model=request.model,
            choices=[],
            usage=ChatCompletionUsage(
                prompt_tokens=10,
                completion_tokens=0,
                total_tokens=10,
            ),
        )


@pytest.mark.asyncio
async def test_complete():
    cognition = CognitionAdapter(FakeChatCompletionService())
    response = await cognition.complete(
        system_prompt="You are helpful.",
        user_prompt="Say hello.",
    )
    assert response.content == "hello"
    assert response.model == ModelRole.PRIMARY_CODER.value
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_critique_uses_judge():
    cognition = CognitionAdapter(FakeChatCompletionService())
    response = await cognition.critique(
        system_prompt="Judge.",
        user_prompt="Review patch.",
    )
    assert response.model == ModelRole.JUDGE.value


@pytest.mark.asyncio
async def test_empty_choices_raises():
    cognition = CognitionAdapter(EmptyChoicesService())
    with pytest.raises(RuntimeError, match="Empty choices"):
        await cognition.complete(
            system_prompt="sys",
            user_prompt="user",
        )


@pytest.mark.asyncio
async def test_complete_custom_role():
    cognition = CognitionAdapter(FakeChatCompletionService())
    response = await cognition.complete(
        system_prompt="sys",
        user_prompt="user",
        model_role=ModelRole.REPO_SYNTHESIZER,
    )
    assert response.model == ModelRole.REPO_SYNTHESIZER.value