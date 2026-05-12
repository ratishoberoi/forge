from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi.testclient import TestClient

from backend.api.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from backend.app import create_app


class StubChatCompletionService:
    async def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="chatcmpl-test",
            object="chat.completion",
            created=1,
            model=request.model or "deepseek-coder",
            choices=[],
            usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        )

    async def stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        yield (
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
            '"created":1,"model":"deepseek-coder","choices":[{"index":0,'
            '"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
            '"created":1,"model":"deepseek-coder","choices":[{"index":0,'
            '"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
        )
        yield "data: [DONE]\n\n"


def test_chat_completion_response_shape() -> None:
    app = create_app(start_runtime=False)
    app.state.chat_service = StubChatCompletionService()
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Write a function."}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["usage"]["total_tokens"] == 13


def test_chat_completion_streaming() -> None:
    app = create_app(start_runtime=False)
    app.state.chat_service = StubChatCompletionService()
    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Write a function."}],
            "stream": True,
        },
    ) as response:
        chunks = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert chunks[-1] == "data: [DONE]"
    payload = json.loads(chunks[0].removeprefix("data: "))
    assert payload["object"] == "chat.completion.chunk"
