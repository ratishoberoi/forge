"""Internal async runtime event bus."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from enum import StrEnum

from pydantic import BaseModel, Field


class EventType(StrEnum):
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    PATCH_PROPOSED = "patch_proposed"
    CRITIQUE_GENERATED = "critique_generated"
    EXECUTION_FINISHED = "execution_finished"
    ORCHESTRATION_TRACE = "orchestration_trace"


class RuntimeEvent(BaseModel):
    id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex}")
    type: EventType
    task_id: str | None = None
    agent_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    payload: dict[str, object] = Field(default_factory=dict)


EventHandler = Callable[[RuntimeEvent], Awaitable[None]]


class AsyncEventBus:
    def __init__(self, *, max_queue_size: int = 1024) -> None:
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue(maxsize=max_queue_size)
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        self._running = False
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
            self._dispatcher_task = None

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    async def publish(self, event: RuntimeEvent) -> None:
        await self._queue.put(event)

    async def _dispatch_loop(self) -> None:
        while self._running:
            event = await self._queue.get()
            handlers = list(self._subscribers.get(event.type, []))
            if handlers:
                await asyncio.gather(*(handler(event) for handler in handlers))
            self._queue.task_done()
