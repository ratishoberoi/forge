"""Mock agents for orchestration verification."""

from __future__ import annotations

import time
import uuid

from backend.repointel.service import RepositoryIntelligenceEngine
from backend.runtime.agents import AgentCapability, AgentExecutionContext, AgentMetadata, BaseAgent
from backend.runtime.messages import (
    AgentMessage,
    CodePatchPayload,
    ContextPayload,
    CritiquePayload,
    MessageKind,
    PlannerOutputPayload,
    TaskRequestPayload,
)
from backend.runtime.tasks import Task


class MockPlannerAgent(BaseAgent):
    def __init__(self, repo_intelligence: RepositoryIntelligenceEngine) -> None:
        super().__init__(
            AgentMetadata(
                name="mock-planner",
                role="planner",
                capabilities=[AgentCapability(name="planning", description="Builds execution plans.")],
            )
        )
        self._repo = repo_intelligence

    async def run(self, task: Task, context: AgentExecutionContext) -> AgentMessage:
        payload = task.payload
        assert isinstance(payload, TaskRequestPayload)
        plan = await self._repo.plan_changes(payload.objective)
        return AgentMessage(
            id=f"msg-{uuid.uuid4().hex}",
            kind=MessageKind.PLANNER_OUTPUT,
            sender=self.metadata.id,
            recipient=None,
            task_id=task.id,
            correlation_id=task.id,
            created_at=time.time(),
            payload=PlannerOutputPayload(
                impacted_files=plan.impacted_files,
                steps=[step.description for step in plan.steps],
                dependency_risks=plan.dependency_risks,
            ),
        )


class MockCoderAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            AgentMetadata(
                name="mock-coder",
                role="coder",
                capabilities=[AgentCapability(name="patching", description="Produces structured code patches.")],
            )
        )

    async def run(self, task: Task, context: AgentExecutionContext) -> AgentMessage:
        payload = task.payload
        target_files = getattr(payload, "target_files", []) if hasattr(payload, "target_files") else []
        diff = "\n".join(
            [
                "--- a/example.py",
                "+++ b/example.py",
                "@@",
                "+# TODO: implement runtime integration",
            ]
        )
        return AgentMessage(
            id=f"msg-{uuid.uuid4().hex}",
            kind=MessageKind.CODE_PATCH,
            sender=self.metadata.id,
            recipient=None,
            task_id=task.id,
            correlation_id=task.id,
            created_at=time.time(),
            payload=CodePatchPayload(
                patch_id=f"patch-{uuid.uuid4().hex}",
                unified_diff=diff,
                impacted_files=target_files,
                risk="medium",
                rationale="Mock coder proposed a placeholder integration patch.",
            ),
        )


class MockCriticAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            AgentMetadata(
                name="mock-critic",
                role="critic",
                capabilities=[AgentCapability(name="critique", description="Reviews candidate outputs.")],
            )
        )

    async def run(self, task: Task, context: AgentExecutionContext) -> AgentMessage:
        shared = await context.shared_context.get_context(task.id)
        summary = shared.summary if shared is not None else task.title
        return AgentMessage(
            id=f"msg-{uuid.uuid4().hex}",
            kind=MessageKind.CRITIQUE,
            sender=self.metadata.id,
            recipient=None,
            task_id=task.id,
            correlation_id=task.id,
            created_at=time.time(),
            payload=CritiquePayload(
                verdict="revise",
                summary=f"Review pending for {summary}.",
                issues=["Mock critic requires execution validation before approval."],
                impacted_files=[],
            ),
        )


class MockContextAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            AgentMetadata(
                name="mock-context",
                role="context",
                capabilities=[AgentCapability(name="context", description="Packages shared context.")],
            )
        )

    async def run(self, task: Task, context: AgentExecutionContext) -> AgentMessage:
        shared = await context.shared_context.get_context(task.id)
        repository_context = []
        if shared and shared.repository_context:
            repository_context = shared.repository_context.related_files
        return AgentMessage(
            id=f"msg-{uuid.uuid4().hex}",
            kind=MessageKind.CONTEXT,
            sender=self.metadata.id,
            recipient=None,
            task_id=task.id,
            correlation_id=task.id,
            created_at=time.time(),
            payload=ContextPayload(
                summary=shared.summary if shared else task.title,
                repository_context=repository_context,
                memory_snapshots=[snapshot.value for snapshot in shared.snapshots] if shared else [],
                token_budget=shared.estimated_tokens if shared else 0,
            ),
        )
