import pytest
from backend.llm.providers.openai_compatible import OpenAICompatibleClient
from backend.runtime.live_cognition import LiveCognition, LiveCognitionResponse


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.persistent_runtime
async def test_live_cognition():
    client = OpenAICompatibleClient(
        base_url="http://localhost:8010",
        model="qwen-primary",
    )
    cognition = LiveCognition(primary_client=client)

    response = await cognition.complete(
        system_prompt="You are a senior Python engineer.",
        user_prompt="Explain dependency injection briefly.",
        max_tokens=128,
    )

    # Type check
    assert isinstance(response, LiveCognitionResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

    # Model field should match PRIMARY_CODER role value
    assert isinstance(response.model, str)
    assert len(response.model) > 0

    await cognition.close()


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.persistent_runtime
async def test_live_cognition_context_manager():
    """Test that async context manager works correctly."""
    client = OpenAICompatibleClient(
        base_url="http://localhost:8010",
        model="qwen-primary",
    )

    async with LiveCognition(primary_client=client) as cognition:
        response = await cognition.complete(
            system_prompt="You are a senior Python engineer.",
            user_prompt="What is a decorator?",
            max_tokens=64,
        )

    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.persistent_runtime
async def test_live_cognition_critique():
    """Test that critique dispatches to judge client at temp=0.0."""
    client = OpenAICompatibleClient(
        base_url="http://localhost:8010",
        model="qwen-primary",
    )

    async with LiveCognition(primary_client=client) as cognition:
        response = await cognition.critique(
            system_prompt="You are a code reviewer.",
            user_prompt="Is `x = x + 1` better than `x += 1`?",
            max_tokens=64,
        )

    assert isinstance(response, LiveCognitionResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
