"""Runtime diagnostics and timing models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentLatencyMetrics(BaseModel):
    agent_id: str
    task_count: int = 0
    total_latency_ms: float = 0.0
    average_latency_ms: float = 0.0


class TaskExecutionMetrics(BaseModel):
    task_id: str
    agent_id: str | None = None
    status: str
    latency_ms: float = 0.0
    queue_wait_ms: float = 0.0
    token_usage: int = 0


class ConcurrencyDiagnostics(BaseModel):
    max_concurrency: int
    active_tasks: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0


class RuntimeDiagnosticsSnapshot(BaseModel):
    orchestration_id: str
    agent_metrics: list[AgentLatencyMetrics] = Field(default_factory=list)
    task_metrics: list[TaskExecutionMetrics] = Field(default_factory=list)
    concurrency: ConcurrencyDiagnostics
