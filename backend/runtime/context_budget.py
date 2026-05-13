from __future__ import annotations
from dataclasses import dataclass

_APPROX_CHARS_PER_TOKEN = 4


@dataclass(slots=True)
class ContextChunk:
    content: str
    priority: int = 0


class ContextBudgetManager:
    """
    Handles prompt context budgeting.
    Responsibilities:
    - approximate token estimation
    - truncation
    - priority ordering
    """

    def __init__(self, max_tokens: int = 6000) -> None:
        self.max_tokens = max_tokens

    def estimate_tokens(self, text: str) -> int:
        """Approximate token count."""
        return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)

    def build_context(self, chunks: list[ContextChunk]) -> str:
        """Build bounded context string."""
        ordered = sorted(chunks, key=lambda c: c.priority, reverse=True)
        result: list[str] = []
        used_tokens = 0
        for chunk in ordered:
            estimated = self.estimate_tokens(chunk.content)
            if used_tokens + estimated > self.max_tokens:
                continue  # skip oversized chunk, try smaller ones
            result.append(chunk.content)
            used_tokens += estimated
        return "\n\n".join(result)