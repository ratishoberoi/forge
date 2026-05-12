"""Token decoding and streaming-safe text assembly."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

_ARTIFACT_TABLE = str.maketrans({"Ġ": " ", "Ċ": "\n"})


def normalize_decoded_text(text: str) -> str:
    return text.translate(_ARTIFACT_TABLE).replace("\u0000", "")


def decode_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
    *,
    clean_up_spaces: bool,
) -> str:
    return normalize_decoded_text(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=clean_up_spaces,
        )
    )


class UtfSafeStreamAssembler:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer
        self._emitted_text = ""

    def push(self, token_ids: list[int]) -> str:
        decoded = decode_token_ids(
            self._tokenizer,
            token_ids,
            clean_up_spaces=False,
        )
        if decoded.startswith(self._emitted_text):
            delta = decoded[len(self._emitted_text) :]
            self._emitted_text = decoded
            return delta

        if len(decoded) > len(self._emitted_text):
            delta = decoded[len(self._emitted_text) :]
            self._emitted_text = decoded
            return delta

        self._emitted_text = decoded
        return ""

    def final_text(self) -> str:
        return normalize_decoded_text(self._emitted_text).strip("\u0000")
