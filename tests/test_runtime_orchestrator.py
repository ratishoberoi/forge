from __future__ import annotations

import asyncio

from backend.runtime.agents import AgentCapability, AgentExecutionContext, AgentMetadata, BaseAgent
from backend.runtime.events import EventType
from backend.runtime.messages import AgentMessage, ContextPayload, MessageKind, TaskRequestPayload
from backend.runtime.orchestrator import Orchestrator
from backend.runtime.tasks import Task, TaskPriority, TaskStatus


class StubAgent(BaseAgent):
    def __init__(self, capability: str, role: str = "worker") -> None:
        super().__init__(
            AgentMetadata(
                name=f"stub-{capability}",
                role=role,
                capabilities=[AgentCapability(name=capability, description="stub")],
            )
        )

    async def run(self, task: Task, context: AgentExecutionContext) -> AgentMessage:
        return AgentMessage(
            id=f"msg-{task.id}",
            kind=MessageKind.CONTEXT,
            sender=self.metadata.id,
            recipient=None,
            task_id=task.id,
            correlation_id=task.id,
            created_at=task.created_at,
            payload=ContextPayload(summary=task.title),
        )


class SlowAgent(StubAgent):
    async def run(self, task: Task, context: AgentExecutionContext) -> AgentMessage:
        await asyncio.sleep(0.05)
        return await super().run(task, context)


async def collect_event(event, sink):
    sink.append(event.type)


def test_orchestrator_executes_dependent_tasks() -> None:
    async def scenario() -> None:
        orchestrator = Orchestrator()
        first = StubAgent("planning")
        second = StubAgent("patching")
        orchestrator.register_agent(first)
        orchestrator.register_agent(second)
        events = []
        orchestrator.event_bus.subscribe(
            EventType.TASK_COMPLETED,
            lambda event: collect_event(event, events),
        )
        await orchestrator.start()
        try:
            task_a = Task(
                title="plan",
                capability="planning",
                priority=TaskPriority.HIGH,
                payload=TaskRequestPayload(objective="plan"),
            )
            task_b = Task(
                title="patch",
                capability="patching",
                dependency_ids=[task_a.id],
                payload=TaskRequestPayload(objective="patch"),
            )
            await orchestrator.submit(task_a)
            await orchestrator.submit(task_b)
            result_a = await orchestrator.wait_for_task(task_a.id)
            result_b = await orchestrator.wait_for_task(task_b.id)
            assert result_a.status == TaskStatus.COMPLETED
            assert result_b.status == TaskStatus.COMPLETED
            assert EventType.TASK_COMPLETED in events
        finally:
            await orchestrator.stop()

    asyncio.run(scenario())


def test_orchestrator_cancels_task() -> None:
    async def scenario() -> None:
        orchestrator = Orchestrator()
        agent = SlowAgent("planning")
        orchestrator.register_agent(agent)
        await orchestrator.start()
        try:
            task = Task(
                title="cancel me",
                capability="planning",
                payload=TaskRequestPayload(objective="cancel"),
            )
            await orchestrator.submit(task)
            await asyncio.sleep(0.01)
            await orchestrator.cancel(task.id)
            result = await orchestrator.wait_for_task(task.id)
            assert result.status == TaskStatus.CANCELLED
        finally:
            await orchestrator.stop()

    asyncio.run(scenario())
