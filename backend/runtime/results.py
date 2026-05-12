"""Aggregation of multi-agent candidate outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.runtime.messages import AgentMessage


class CandidateSolution(BaseModel):
    task_id: str
    agent_id: str
    message: AgentMessage
    score: float = 0.0
    confidence: float = 0.0
    metadata: dict[str, object] = Field(default_factory=dict)


class AggregatedResult(BaseModel):
    task_id: str
    candidates: list[CandidateSolution] = Field(default_factory=list)
    winner: CandidateSolution | None = None


class ResultAggregator:
    def __init__(self) -> None:
        self._results: dict[str, list[CandidateSolution]] = {}

    def add(self, candidate: CandidateSolution) -> None:
        self._results.setdefault(candidate.task_id, []).append(candidate)

    def aggregate(self, task_id: str) -> AggregatedResult:
        candidates = list(self._results.get(task_id, []))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        winner = candidates[0] if candidates else None
        return AggregatedResult(task_id=task_id, candidates=candidates, winner=winner)
