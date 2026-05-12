"""Async orchestration engine."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field

from backend.config.settings import Settings, get_settings
from backend.core.errors import ConfigurationError, InvalidRequestError
from backend.core.logging import log_event
from backend.runtime.agents import AgentExecutionContext, BaseAgent
from backend.runtime.context import SharedContextStore
from backend.runtime.diagnostics import (
    AgentLatencyMetrics,
    ConcurrencyDiagnostics,
    RuntimeDiagnosticsSnapshot,
    TaskExecutionMetrics,
)
from backend.runtime.events import AsyncEventBus, EventType, RuntimeEvent
from backend.runtime.results import CandidateSolution, ResultAggregator
from backend.runtime.tasks import Task, TaskStatus

logger = logging.getLogger(__name__)


@dataclass(order=True, slots=True)
class QueueItem:
    priority: int
    sequence: int
    task_id: str = field(compare=False)


class Orchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        shared_context: SharedContextStore | None = None,
        event_bus: AsyncEventBus | None = None,
        result_aggregator: ResultAggregator | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._shared_context = shared_context or SharedContextStore(self._settings)
        self._event_bus = event_bus or AsyncEventBus(
            max_queue_size=self._settings.runtime_event_queue_size
        )
        self._aggregator = result_aggregator or ResultAggregator()
        self._agents: dict[str, BaseAgent] = {}
        self._tasks: dict[str, Task] = {}
        self._active: dict[str, asyncio.Task[None]] = {}
        self._queue: asyncio.PriorityQueue[QueueItem] = asyncio.PriorityQueue()
        self._sequence = 0
        self._runner_task: asyncio.Task[None] | None = None
        self._semaphore = asyncio.Semaphore(self._settings.runtime_max_concurrency)
        self._running = False
        self._orchestration_id = f"orch-{uuid.uuid4().hex}"
        self._diagnostics: dict[str, TaskExecutionMetrics] = {}
        self._agent_metrics: dict[str, AgentLatencyMetrics] = {}

    @property
    def event_bus(self) -> AsyncEventBus:
        return self._event_bus

    @property
    def shared_context(self) -> SharedContextStore:
        return self._shared_context

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents[agent.metadata.id] = agent

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._event_bus.start()
        for agent in self._agents.values():
            await agent.on_start()
        self._runner_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._runner_task is not None:
            self._runner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner_task
            self._runner_task = None
        for active_task in list(self._active.values()):
            active_task.cancel()
        if self._active:
            await asyncio.gather(*self._active.values(), return_exceptions=True)
        for agent in self._agents.values():
            await agent.on_stop()
        await self._event_bus.stop()

    async def submit(self, task: Task) -> str:
        if task.agent is None and task.capability is None:
            raise InvalidRequestError("Task must specify either an agent or capability.")
        if task.max_retries == 0:
            task.max_retries = self._settings.runtime_max_retries
        if task.timeout_ms is None:
            task.timeout_ms = self._settings.runtime_default_timeout_ms
        if task.dependency_ids:
            task.status = TaskStatus.WAITING
        self._tasks[task.id] = task
        await self._enqueue_if_ready(task)
        return task.id

    async def cancel(self, task_id: str, reason: str = "Cancelled by request.") -> None:
        task = self._tasks.get(task_id)
        if task is None:
            raise InvalidRequestError(f"Task '{task_id}' does not exist.")
        task.mark_cancelled(reason)
        running = self._active.get(task_id)
        if running is not None:
            running.cancel()
        await self._event_bus.publish(
            RuntimeEvent(type=EventType.TASK_CANCELLED, task_id=task_id, payload={"reason": reason})
        )

    async def wait_for_task(self, task_id: str) -> Task:
        while True:
            task = self._tasks[task_id]
            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMED_OUT,
            }:
                return task
            await asyncio.sleep(self._settings.runtime_agent_heartbeat_ms / 1000)

    def diagnostics(self) -> RuntimeDiagnosticsSnapshot:
        return RuntimeDiagnosticsSnapshot(
            orchestration_id=self._orchestration_id,
            agent_metrics=list(self._agent_metrics.values()),
            task_metrics=list(self._diagnostics.values()),
            concurrency=ConcurrencyDiagnostics(
                max_concurrency=self._settings.runtime_max_concurrency,
                active_tasks=len(self._active),
                queued_tasks=self._queue.qsize(),
                completed_tasks=sum(
                    1 for task in self._tasks.values() if task.status == TaskStatus.COMPLETED
                ),
            ),
        )

    async def _enqueue_if_ready(self, task: Task) -> None:
        if task.status == TaskStatus.CANCELLED:
            return
        unresolved = [
            dependency_id
            for dependency_id in task.dependency_ids
            if self._tasks.get(dependency_id) is None
            or self._tasks[dependency_id].status != TaskStatus.COMPLETED
        ]
        if unresolved:
            task.status = TaskStatus.WAITING
            return
        task.status = TaskStatus.PENDING
        self._sequence += 1
        await self._queue.put(
            QueueItem(priority=int(task.priority), sequence=self._sequence, task_id=task.id)
        )

    async def _run_loop(self) -> None:
        while self._running:
            item = await self._queue.get()
            task = self._tasks.get(item.task_id)
            if task is None or task.status in {
                TaskStatus.CANCELLED,
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
            }:
                self._queue.task_done()
                continue
            run_task = asyncio.create_task(self._execute_with_accounting(task))
            self._active[task.id] = run_task
            self._queue.task_done()

    async def _execute_with_accounting(self, task: Task) -> None:
        await self._semaphore.acquire()
        try:
            await self._execute_task(task)
        finally:
            self._active.pop(task.id, None)
            self._semaphore.release()

    async def _execute_task(self, task: Task) -> None:
        agent = self._resolve_agent(task)
        task.mark_started()
        queue_wait_ms = (task.started_at - task.created_at) * 1000 if task.started_at else 0.0
        await self._event_bus.publish(
            RuntimeEvent(
                type=EventType.TASK_STARTED,
                task_id=task.id,
                agent_id=agent.metadata.id,
                payload={"title": task.title},
            )
        )
        log_event(
            logger,
            logging.INFO,
            "runtime.task.started",
            "Runtime task started.",
            orchestration_id=self._orchestration_id,
            task_id=task.id,
            agent_id=agent.metadata.id,
            queue_wait_ms=round(queue_wait_ms, 3),
        )
        started = time.perf_counter()
        try:
            await self._shared_context.build_context(
                task.id,
                query=getattr(task.payload, "objective", task.title),
                summary=task.title,
            )
            context = AgentExecutionContext(
                orchestration_id=self._orchestration_id,
                shared_context=self._shared_context,
            )
            result = await asyncio.wait_for(
                agent.execute(task, context),
                timeout=(task.timeout_ms or self._settings.runtime_default_timeout_ms) / 1000,
            )
            task.mark_completed(result)
            latency_ms = (time.perf_counter() - started) * 1000
            self._record_task_metrics(task, agent.metadata.id, latency_ms, queue_wait_ms)
            self._aggregator.add(
                CandidateSolution(
                    task_id=task.id,
                    agent_id=agent.metadata.id,
                    message=result,
                    score=1.0,
                    confidence=0.5,
                )
            )
            await self._publish_result_event(task, result, agent.metadata.id)
            await self._release_waiting_dependencies()
        except asyncio.TimeoutError:
            await self._handle_failure(task, agent.metadata.id, "Task timed out.", timed_out=True)
        except asyncio.CancelledError:
            task.mark_cancelled("Task cancelled during execution.")
            raise
        except Exception as exc:
            await self._handle_failure(task, agent.metadata.id, str(exc), timed_out=False)

    def _resolve_agent(self, task: Task) -> BaseAgent:
        if task.agent is not None:
            agent = self._agents.get(task.agent)
            if agent is None:
                raise ConfigurationError(f"Agent '{task.agent}' is not registered.")
            return agent
        for agent in self._agents.values():
            if any(capability.name == task.capability for capability in agent.metadata.capabilities):
                return agent
        raise ConfigurationError(f"No registered agent satisfies capability '{task.capability}'.")

    async def _handle_failure(
        self,
        task: Task,
        agent_id: str,
        error: str,
        *,
        timed_out: bool,
    ) -> None:
        if task.retries < task.max_retries:
            task.retries += 1
            task.reset_for_retry()
            await self._enqueue_if_ready(task)
            return
        task.mark_failed(error, timed_out=timed_out)
        latency_ms = 0.0
        if task.started_at is not None:
            latency_ms = (task.completed_at - task.started_at) * 1000 if task.completed_at else 0.0
        self._record_task_metrics(task, agent_id, latency_ms, 0.0)
        event_type = EventType.TASK_FAILED
        await self._event_bus.publish(
            RuntimeEvent(type=event_type, task_id=task.id, agent_id=agent_id, payload={"error": error})
        )
        log_event(
            logger,
            logging.ERROR,
            "runtime.task.failed",
            "Runtime task failed.",
            orchestration_id=self._orchestration_id,
            task_id=task.id,
            agent_id=agent_id,
            error=error,
            timed_out=timed_out,
        )
        await self._release_waiting_dependencies()

    def _record_task_metrics(
        self,
        task: Task,
        agent_id: str,
        latency_ms: float,
        queue_wait_ms: float,
    ) -> None:
        metric = TaskExecutionMetrics(
            task_id=task.id,
            agent_id=agent_id,
            status=task.status.value,
            latency_ms=round(latency_ms, 3),
            queue_wait_ms=round(queue_wait_ms, 3),
            token_usage=self._infer_token_usage(task),
        )
        self._diagnostics[task.id] = metric
        agent_metric = self._agent_metrics.setdefault(
            agent_id,
            AgentLatencyMetrics(agent_id=agent_id),
        )
        agent_metric.task_count += 1
        agent_metric.total_latency_ms += metric.latency_ms
        agent_metric.average_latency_ms = round(
            agent_metric.total_latency_ms / max(agent_metric.task_count, 1),
            3,
        )

    async def _publish_result_event(self, task: Task, result, agent_id: str) -> None:
        event_type = EventType.TASK_COMPLETED
        if result.kind.value == "code_patch":
            event_type = EventType.PATCH_PROPOSED
        elif result.kind.value == "critique":
            event_type = EventType.CRITIQUE_GENERATED
        elif result.kind.value == "execution_result":
            event_type = EventType.EXECUTION_FINISHED
        await self._event_bus.publish(
            RuntimeEvent(
                type=event_type,
                task_id=task.id,
                agent_id=agent_id,
                payload={"message_kind": result.kind.value},
            )
        )

    async def _release_waiting_dependencies(self) -> None:
        for task in self._tasks.values():
            if task.status == TaskStatus.WAITING:
                await self._enqueue_if_ready(task)

    @staticmethod
    def _infer_token_usage(task: Task) -> int:
        text = getattr(task.payload, "objective", "") or getattr(task.payload, "summary", "")
        return max(0, len(text) // 4)
