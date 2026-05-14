"""Multi-agent runtime foundation."""

from backend.runtime.agents import AgentCapability, AgentMetadata, BaseAgent
from backend.runtime.api import MultiAgentRuntime
from backend.runtime.binding import BindingDecision, RuntimeBinder, RuntimeBinding
from backend.runtime.candidate import CandidateCollection, PatchCandidate
from backend.runtime.consensus import ConsensusEngine, ConsensusResult
from backend.runtime.convergence import ConvergenceTracker
from backend.runtime.context import SharedContextStore
from backend.runtime.debate import DebateOrchestrator, DebateResult, DebateRound
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
from backend.runtime.routing import CognitionRole, CognitionRouter, ContextBudget, RoutedContext, RoutingDecision
from backend.runtime.ranking import compare_scores, rank_candidates
from backend.runtime.reviewer import CandidateReview, CandidateReviewer
from backend.runtime.tasks import Task, TaskPriority, TaskStatus

__all__ = [
    "AgentCapability",
    "AgentMetadata",
    "AgentMessage",
    "AsyncEventBus",
    "BaseAgent",
    "BindingDecision",
    "CandidateSolution",
    "CandidateCollection",
    "CandidateReview",
    "CandidateReviewer",
    "CodePatchPayload",
    "CognitionRole",
    "CognitionRouter",
    "ConsensusEngine",
    "ConsensusResult",
    "ConvergenceTracker",
    "ContextBudget",
    "ContextPayload",
    "CritiquePayload",
    "DebateOrchestrator",
    "DebateResult",
    "DebateRound",
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
    "RuntimeBinder",
    "RuntimeBinding",
    "RoutedContext",
    "RoutingDecision",
    "build_candidate_comparison_prompt",
    "build_execution_summary",
    "build_convergence_warning_prompt",
    "build_patch_judge_prompt",
    "build_retry_candidate_prompt",
    "build_retry_prompt",
    "build_self_repair_prompt",
    "compare_scores",
    "rank_candidates",
    "summarize_execution_failures",
]
