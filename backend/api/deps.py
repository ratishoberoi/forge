"""FastAPI dependency helpers."""

from fastapi import Request

from backend.llm.service import ChatCompletionService


def get_chat_service(request: Request) -> ChatCompletionService:
    return request.app.state.chat_service
