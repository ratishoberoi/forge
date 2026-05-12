"""Multi-agent runtime foundation."""

from backend.runtime.agents import AgentCapability, AgentMetadata, BaseAgent
from backend.runtime.api import MultiAgentRuntime
from backend.runtime.context import SharedContextStore
from backend.runtime.events import AsyncEventBus, EventType, RuntimeEvent
from backend.runtime.messages import (
    AgentMessage,
    CodePatchPayload,
    ContextPayload,
    CritiquePayload,
    ExecutionResultPayload,
    PlannerOutputPayload,
    TaskRequestPayload,
)
from backend.runtime.orchestrator import Orchestrator
from backend.runtime.patches import PatchBundle, PatchRisk, PatchTarget, StructuredPatch
from backend.runtime.results import CandidateSolution, ResultAggregator
from backend.runtime.tasks import Task, TaskPriority, TaskStatus

__all__ = [
    "AgentCapability",
    "AgentMetadata",
    "AgentMessage",
    "AsyncEventBus",
    "BaseAgent",
    "CandidateSolution",
    "CodePatchPayload",
    "ContextPayload",
    "CritiquePayload",
    "EventType",
    "ExecutionResultPayload",
    "MultiAgentRuntime",
    "Orchestrator",
    "PatchBundle",
    "PatchRisk",
    "PatchTarget",
    "PlannerOutputPayload",
    "ResultAggregator",
    "RuntimeEvent",
    "SharedContextStore",
    "StructuredPatch",
    "Task",
    "TaskPriority",
    "TaskRequestPayload",
    "TaskStatus",
]
