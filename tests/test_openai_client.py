import pytest
from backend.llm.providers.openai_compatible import OpenAICompatibleClient


@pytest.mark.asyncio
async def test_live_generation():
    async with OpenAICompatibleClient(
        base_url="http://localhost:8010",
        model="qwen-primary",
    ) as client:
        response = await client.generate(
            system_prompt="You are a senior engineer.",
            user_prompt="Say hello in one sentence.",
            max_tokens=32,
        )
        assert isinstance(response, str)
        assert len(response) > 0


@pytest.mark.asyncio
async def test_empty_choices_raises():
    """Guard against malformed responses with empty choices."""
    import httpx
    import respx

    async with respx.mock:
        respx.post("http://localhost:8010/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        async with OpenAICompatibleClient(
            base_url="http://localhost:8010",
            model="qwen-primary",
        ) as client:
            with pytest.raises(RuntimeError, match="Empty choices"):
                await client.generate(
                    system_prompt="sys",
                    user_prompt="user",
                )


@pytest.mark.asyncio
async def test_http_error_raises():
    """Non-2xx response must raise."""
    import httpx
    import respx

    async with respx.mock:
        respx.post("http://localhost:8010/v1/chat/completions").mock(
            return_value=httpx.Response(500)
        )
        async with OpenAICompatibleClient(
            base_url="http://localhost:8010",
            model="qwen-primary",
        ) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.generate(
                    system_prompt="sys",
                    user_prompt="user",
                )