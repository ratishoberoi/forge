import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.runtime.live_adapter import LiveCognitionAdapter, LiveAdapterError
from backend.runtime.runtime_client import RuntimeClient
from backend.runtime.runtime_registry import RuntimeEndpoint


# ── Unit tests (no server needed) ───────────────────────────────────────────

def make_mock_client(response: str = "mocked response") -> RuntimeClient:
    client = MagicMock(spec=RuntimeClient)
    client.chat_completion = AsyncMock(return_value=response)
    client.health_check = AsyncMock(return_value=True)
    return client


@pytest.mark.asyncio
async def test_execute_calls_client_with_correct_args():
    mock_client = make_mock_client("def hello(): pass")
    adapter = LiveCognitionAdapter(client=mock_client)

    response = await adapter.execute(
        role="PRIMARY_CODER",
        prompt="Write a Python hello world function.",
    )

    assert response == "def hello(): pass"
    mock_client.chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_with_system_prompt_builds_correct_messages():
    mock_client = make_mock_client("ok")
    adapter = LiveCognitionAdapter(client=mock_client)

    await adapter.execute(
        role="PRIMARY_CODER",
        prompt="Write a function.",
        system_prompt="You are a senior Python engineer.",
    )

    call_kwargs = mock_client.chat_completion.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_execute_unknown_role_raises_live_adapter_error():
    adapter = LiveCognitionAdapter(client=make_mock_client())

    with pytest.raises(LiveAdapterError, match="Unknown runtime role"):
        await adapter.execute(role="NONEXISTENT", prompt="hello")


@pytest.mark.asyncio
async def test_execute_with_endpoint_bypasses_registry():
    mock_client = make_mock_client("direct response")
    adapter = LiveCognitionAdapter(client=mock_client)

    endpoint = RuntimeEndpoint(
        role="TEST",
        model_name="test-model",
        base_url="http://localhost:9999",
    )

    response = await adapter.execute_with_endpoint(
        endpoint=endpoint,
        prompt="hello",
    )

    assert response == "direct response"
    call_kwargs = mock_client.chat_completion.call_args.kwargs
    assert call_kwargs["base_url"] == "http://localhost:9999"
    assert call_kwargs["model"] == "test-model"


@pytest.mark.asyncio
async def test_health_check_returns_true_when_reachable():
    mock_client = make_mock_client()
    adapter = LiveCognitionAdapter(client=mock_client)

    result = await adapter.health_check("PRIMARY_CODER")
    assert result is True


@pytest.mark.asyncio
async def test_health_check_returns_false_for_unknown_role():
    adapter = LiveCognitionAdapter(client=make_mock_client())
    result = await adapter.health_check("NONEXISTENT")
    assert result is False


def test_build_messages_without_system_prompt():
    messages = LiveCognitionAdapter._build_messages("hello", None)
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "hello"}


def test_build_messages_with_system_prompt():
    messages = LiveCognitionAdapter._build_messages("hello", "you are an expert")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


# ── Live tests (requires running inference server) ───────────────────────────

@pytest.mark.asyncio
@pytest.mark.live
async def test_live_primary_runtime():
    adapter = LiveCognitionAdapter()
    response = await adapter.execute(
        role="PRIMARY_CODER",
        prompt="Write a Python hello world function.",
        system_prompt="You are a senior Python engineer.",
    )
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
@pytest.mark.live
async def test_live_judge_runtime():
    adapter = LiveCognitionAdapter()
    response = await adapter.execute(
        role="JUDGE",
        prompt="Critique whether type hints change Python runtime behavior.",
        system_prompt="You are a strict code reviewer.",
    )
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
@pytest.mark.live
async def test_live_health_check_primary():
    adapter = LiveCognitionAdapter()
    result = await adapter.health_check("PRIMARY_CODER")
    assert isinstance(result, bool)