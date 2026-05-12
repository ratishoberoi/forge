"""Task lifecycle models."""

from __future__ import annotations

import time
import uuid
from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field

from backend.runtime.messages import AgentMessage, PayloadType


class TaskStatus(StrEnum):
    PENDING = "pending"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class TaskPriority(IntEnum):
    LOW = 30
    NORMAL = 20
    HIGH = 10
    CRITICAL = 0


class Task(BaseModel):
    id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex}")
    title: str
    agent: str | None = None
    capability: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    retries: int = Field(default=0, ge=0)
    max_retries: int = Field(default=0, ge=0)
    dependency_ids: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    deadline_at: float | None = None
    timeout_ms: int | None = Field(default=None, ge=1)
    payload: PayloadType
    messages: list[AgentMessage] = Field(default_factory=list)
    result: AgentMessage | None = None
    error: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    def mark_started(self) -> None:
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

    def mark_completed(self, result: AgentMessage) -> None:
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result

    def mark_failed(self, error: str, *, timed_out: bool = False) -> None:
        self.status = TaskStatus.TIMED_OUT if timed_out else TaskStatus.FAILED
        self.completed_at = time.time()
        self.error = error

    def mark_cancelled(self, reason: str) -> None:
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()
        self.error = reason

    def reset_for_retry(self) -> None:
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.result = None
