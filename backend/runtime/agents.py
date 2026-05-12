"""Base agent abstraction."""

from __future__ import annotations

import abc
import logging
import time
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field

from backend.core.logging import log_event
from backend.runtime.messages import AgentMessage
from backend.runtime.tasks import Task

logger = logging.getLogger(__name__)


class AgentCapability(BaseModel):
    name: str
    description: str


class AgentMetadata(BaseModel):
    id: str = Field(default_factory=lambda: f"agent-{uuid.uuid4().hex}")
    name: str
    role: str
    capabilities: list[AgentCapability] = Field(default_factory=list)
    version: str = "0.1.0"


@dataclass(slots=True)
class AgentExecutionContext:
    orchestration_id: str
    shared_context: "SharedContextStore"


class BaseAgent(abc.ABC):
    def __init__(self, metadata: AgentMetadata) -> None:
        self.metadata = metadata

    async def on_start(self) -> None:
        log_event(
            logger,
            logging.INFO,
            "runtime.agent.started",
            "Agent started.",
            agent_id=self.metadata.id,
            role=self.metadata.role,
        )

    async def on_stop(self) -> None:
        log_event(
            logger,
            logging.INFO,
            "runtime.agent.stopped",
            "Agent stopped.",
            agent_id=self.metadata.id,
            role=self.metadata.role,
        )

    async def before_execute(self, task: Task) -> None:
        log_event(
            logger,
            logging.INFO,
            "runtime.agent.task.before",
            "Agent preparing task.",
            agent_id=self.metadata.id,
            task_id=task.id,
            task_title=task.title,
        )

    async def after_execute(self, task: Task, result: AgentMessage, latency_ms: float) -> None:
        log_event(
            logger,
            logging.INFO,
            "runtime.agent.task.after",
            "Agent completed task.",
            agent_id=self.metadata.id,
            task_id=task.id,
            latency_ms=round(latency_ms, 3),
        )

    async def execute(self, task: Task, context: AgentExecutionContext) -> AgentMessage:
        started = time.perf_counter()
        await self.before_execute(task)
        result = await self.run(task, context)
        latency_ms = (time.perf_counter() - started) * 1000
        await self.after_execute(task, result, latency_ms)
        return result

    @abc.abstractmethod
    async def run(self, task: Task, context: AgentExecutionContext) -> AgentMessage:
        raise NotImplementedError


from backend.runtime.context import SharedContextStore  # noqa: E402
