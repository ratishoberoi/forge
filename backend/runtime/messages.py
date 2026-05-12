"""Typed agent message protocol."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class MessageKind(StrEnum):
    TASK_REQUEST = "task_request"
    CONTEXT = "context"
    CODE_PATCH = "code_patch"
    CRITIQUE = "critique"
    EXECUTION_RESULT = "execution_result"
    PLANNER_OUTPUT = "planner_output"


class TaskRequestPayload(BaseModel):
    objective: str
    constraints: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)


class ContextPayload(BaseModel):
    summary: str
    repository_context: list[str] = Field(default_factory=list)
    memory_snapshots: list[str] = Field(default_factory=list)
    token_budget: int = Field(default=0, ge=0)


class CodePatchPayload(BaseModel):
    patch_id: str
    unified_diff: str
    impacted_files: list[str] = Field(default_factory=list)
    risk: str = "unknown"
    rationale: str | None = None


class CritiquePayload(BaseModel):
    verdict: Literal["approve", "revise", "reject"]
    summary: str
    issues: list[str] = Field(default_factory=list)
    impacted_files: list[str] = Field(default_factory=list)


class ExecutionResultPayload(BaseModel):
    success: bool
    command: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None


class PlannerOutputPayload(BaseModel):
    impacted_files: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    dependency_risks: list[str] = Field(default_factory=list)


PayloadType = (
    TaskRequestPayload
    | ContextPayload
    | CodePatchPayload
    | CritiquePayload
    | ExecutionResultPayload
    | PlannerOutputPayload
)


class AgentMessage(BaseModel):
    id: str
    kind: MessageKind
    sender: str
    recipient: str | None = None
    task_id: str | None = None
    correlation_id: str | None = None
    payload: PayloadType
    created_at: float
