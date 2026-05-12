"""Chat completion orchestration over vLLM."""

from __future__ import annotations

import logging
import json
import time
import uuid
from collections.abc import AsyncGenerator

from vllm import SamplingParams
from vllm.outputs import RequestOutput

from backend.api.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from backend.config.settings import Settings
from backend.core.logging import log_event
from backend.core.tracing import RequestTrace
from backend.core.errors import InvalidRequestError, ModelNotReadyError
from backend.llm.decoding import UtfSafeStreamAssembler, decode_token_ids
from backend.llm.engine import LLMEngineManager
from backend.llm.prompting import render_chat_prompt
from backend.llm.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ChatCompletionService:
    def __init__(self, settings: Settings, engine_manager: LLMEngineManager) -> None:
        self._settings = settings
        self._engine_manager = engine_manager
        self._registry = ModelRegistry(settings)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        engine = self._engine_manager.engine
        tokenizer = self._engine_manager.tokenizer
        if engine is None or tokenizer is None:
            raise ModelNotReadyError()

        model_record = self._registry.resolve_generation_model(request.model)
        model_name = model_record.alias
        created = int(time.time())
        request_id = request.request_id or self._build_request_id()
        trace = RequestTrace(
            request_id=request_id,
            model=model_name,
            agent_id=request.agent_id,
            stream=False,
        )
        log_event(
            logger,
            logging.INFO,
            "llm.request.started",
            "Starting chat completion request.",
            request_id=request_id,
            agent_id=request.agent_id,
            model=model_name,
            stream=False,
        )
        prompt = render_chat_prompt(tokenizer, request.messages)
        sampling_params = self._build_sampling_params(request)

        final_output: RequestOutput | None = None
        async for output in engine.generate(
            prompt,
            sampling_params,
            request_id,
            prompt_text=prompt,
        ):
            if final_output is None and output.outputs and output.outputs[0].token_ids:
                trace.mark_first_token()
            final_output = output

        if final_output is None or not final_output.outputs:
            raise InvalidRequestError("Model returned no completion.")

        completion = final_output.outputs[0]
        prompt_tokens = len(final_output.prompt_token_ids or [])
        completion_tokens = len(completion.token_ids)
        trace.finish(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        completion_text = decode_token_ids(
            tokenizer,
            list(completion.token_ids),
            clean_up_spaces=True,
        )
        usage = ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        log_event(
            logger,
            logging.INFO,
            "llm.request.completed",
            "Chat completion request finished.",
            **trace.as_log_fields(),
        )
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=created,
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_text),
                    finish_reason=completion.finish_reason,
                )
            ],
            usage=usage,
        )

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        engine = self._engine_manager.engine
        tokenizer = self._engine_manager.tokenizer
        if engine is None or tokenizer is None:
            raise ModelNotReadyError()

        model_record = self._registry.resolve_generation_model(request.model)
        model_name = model_record.alias
        created = int(time.time())
        request_id = request.request_id or self._build_request_id()
        trace = RequestTrace(
            request_id=request_id,
            model=model_name,
            agent_id=request.agent_id,
            stream=True,
        )
        log_event(
            logger,
            logging.INFO,
            "llm.request.started",
            "Starting streaming chat completion request.",
            request_id=request_id,
            agent_id=request.agent_id,
            model=model_name,
            stream=True,
        )
        prompt = render_chat_prompt(tokenizer, request.messages)
        sampling_params = self._build_sampling_params(request)
        assembler = UtfSafeStreamAssembler(tokenizer)

        initial_chunk = ChatCompletionChunk(
            id=request_id,
            object="chat.completion.chunk",
            created=created,
            model=model_name,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield self._format_sse(initial_chunk.model_dump(mode="json"))

        final_output: RequestOutput | None = None
        async for output in engine.generate(
            prompt,
            sampling_params,
            request_id,
            prompt_text=prompt,
        ):
            if not output.outputs:
                continue

            final_output = output
            completion = output.outputs[0]
            delta_text = assembler.push(list(completion.token_ids))
            if delta_text:
                trace.mark_first_token()
                chunk = ChatCompletionChunk(
                    id=request_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionDelta(content=delta_text),
                            finish_reason=None,
                        )
                    ],
                )
                yield self._format_sse(chunk.model_dump(mode="json"))

            if completion.finish_reason is not None:
                final_chunk = ChatCompletionChunk(
                    id=request_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionDelta(),
                            finish_reason=completion.finish_reason,
                        )
                    ],
                )
                yield self._format_sse(final_chunk.model_dump(mode="json"))

        prompt_tokens = len(final_output.prompt_token_ids or []) if final_output else 0
        completion_tokens = len(final_output.outputs[0].token_ids) if final_output and final_output.outputs else 0
        trace.finish(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        log_event(
            logger,
            logging.INFO,
            "llm.request.completed",
            "Streaming chat completion request finished.",
            **trace.as_log_fields(),
        )
        yield "data: [DONE]\n\n"

    def _build_sampling_params(self, request: ChatCompletionRequest) -> SamplingParams:
        return SamplingParams(
            temperature=request.temperature if request.temperature is not None else self._settings.model_temperature,
            top_p=request.top_p if request.top_p is not None else self._settings.model_top_p,
            max_tokens=request.max_tokens if request.max_tokens is not None else self._settings.model_max_tokens,
        )

    @staticmethod
    def _build_request_id() -> str:
        return f"chatcmpl-{uuid.uuid4().hex}"

    @staticmethod
    def _format_sse(payload: dict[str, object]) -> str:
        return f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"
