"""Structured patch representation."""

from __future__ import annotations

import time
import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class PatchRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class PatchTarget(BaseModel):
    path: str
    change_type: str = "modify"
    start_line: int | None = None
    end_line: int | None = None


class StructuredPatch(BaseModel):
    id: str = Field(default_factory=lambda: f"patch-{uuid.uuid4().hex}")
    title: str
    description: str | None = None
    unified_diff: str
    impacted_files: list[PatchTarget] = Field(default_factory=list)
    risk: PatchRisk = PatchRisk.UNKNOWN
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class PatchBundle(BaseModel):
    task_id: str
    agent_id: str
    patches: list[StructuredPatch] = Field(default_factory=list)
    aggregate_risk: PatchRisk = PatchRisk.UNKNOWN
