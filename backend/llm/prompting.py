"""Prompt rendering helpers."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from backend.api.schemas.chat import ChatMessage


def render_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[ChatMessage],
) -> str:
    prompt_messages: list[dict[str, str]] = []
    for message in messages:
        prompt_messages.append({"role": message.role, "content": message.content})

    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
