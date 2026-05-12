"""Request-level tracing models."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class RequestTrace:
    request_id: str
    model: str
    agent_id: str | None
    stream: bool
    started_at: float = field(default_factory=perf_counter)
    first_token_at: float | None = None
    finished_at: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def mark_first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = perf_counter()

    def finish(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.finished_at = perf_counter()

    @property
    def latency_ms(self) -> float:
        finished = self.finished_at if self.finished_at is not None else perf_counter()
        return round((finished - self.started_at) * 1000, 3)

    @property
    def first_token_latency_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started_at) * 1000, 3)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_log_fields(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "model": self.model,
            "stream": self.stream,
            "latency_ms": self.latency_ms,
            "first_token_latency_ms": self.first_token_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
