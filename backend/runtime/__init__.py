"""Multi-agent runtime foundation."""

from backend.runtime.agents import AgentCapability, AgentMetadata, BaseAgent
from backend.runtime.api import MultiAgentRuntime
from backend.runtime.candidate import CandidateCollection, PatchCandidate
from backend.runtime.convergence import ConvergenceTracker
from backend.runtime.context import SharedContextStore
from backend.runtime.execution_review import (
    ExecutionReview,
    ExecutionSeverity,
    build_execution_summary,
    summarize_execution_failures,
)
from backend.runtime.events import AsyncEventBus, EventType, RuntimeEvent
from backend.runtime.judge import JudgeResult, JudgeScore, PatchJudge
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
from backend.runtime.retry import RetryDecision, RetryOrchestrator, RetryPolicy, RetryResult
from backend.runtime.retry_prompting import (
    build_convergence_warning_prompt,
    build_retry_candidate_prompt,
    build_self_repair_prompt,
)
from backend.runtime.review_prompting import (
    build_candidate_comparison_prompt,
    build_patch_judge_prompt,
    build_retry_prompt,
)
from backend.runtime.results import CandidateSolution, ResultAggregator
from backend.runtime.tasks import Task, TaskPriority, TaskStatus

__all__ = [
    "AgentCapability",
    "AgentMetadata",
    "AgentMessage",
    "AsyncEventBus",
    "BaseAgent",
    "CandidateSolution",
    "CandidateCollection",
    "CodePatchPayload",
    "ConvergenceTracker",
    "ContextPayload",
    "CritiquePayload",
    "ExecutionReview",
    "ExecutionSeverity",
    "JudgeResult",
    "JudgeScore",
    "EventType",
    "ExecutionResultPayload",
    "MultiAgentRuntime",
    "Orchestrator",
    "PatchCandidate",
    "PatchJudge",
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
    "RetryDecision",
    "RetryOrchestrator",
    "RetryPolicy",
    "RetryResult",
    "build_candidate_comparison_prompt",
    "build_execution_summary",
    "build_convergence_warning_prompt",
    "build_patch_judge_prompt",
    "build_retry_candidate_prompt",
    "build_retry_prompt",
    "build_self_repair_prompt",
    "summarize_execution_failures",
]
