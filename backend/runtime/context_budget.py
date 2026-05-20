from __future__ import annotations
from dataclasses import dataclass, field

_APPROX_CHARS_PER_TOKEN = 4


@dataclass(slots=True)
class ContextChunk:
    content: str
    priority: int = 0
    label: str = "context"


@dataclass(slots=True)
class ContextBudgetResult:
    context: str
    estimated_tokens: int
    max_tokens: int
    included: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    truncated: bool = False


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
        return self.build(chunks).context

    def build(self, chunks: list[ContextChunk]) -> ContextBudgetResult:
        """Build bounded context string with accounting metadata."""
        ordered = sorted(chunks, key=lambda c: c.priority, reverse=True)
        result: list[str] = []
        used_tokens = 0
        included: list[str] = []
        dropped: list[str] = []
        for chunk in ordered:
            estimated = self.estimate_tokens(chunk.content)
            if used_tokens + estimated > self.max_tokens:
                dropped.append(chunk.label)
                continue  # skip oversized chunk, try smaller ones
            result.append(chunk.content)
            used_tokens += estimated
            included.append(chunk.label)
        return ContextBudgetResult(
            context="\n\n".join(result),
            estimated_tokens=used_tokens,
            max_tokens=self.max_tokens,
            included=included,
            dropped=dropped,
            truncated=bool(dropped),
        )
