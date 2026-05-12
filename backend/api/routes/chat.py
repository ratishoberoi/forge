"""OpenAI-compatible chat completion routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from backend.api.deps import get_chat_service
from backend.api.schemas.chat import ChatCompletionRequest
from backend.llm.service import ChatCompletionService

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    service: ChatCompletionService = Depends(get_chat_service),
) -> Response:
    if request.stream:
        generator = service.stream_chat_completion(request)
        return StreamingResponse(generator, media_type="text/event-stream")

    response = await service.create_chat_completion(request)
    return Response(
        content=response.model_dump_json(),
        media_type="application/json",
    )
