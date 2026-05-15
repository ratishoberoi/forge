from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

from backend.runtime.courtroom_roles import CourtroomRole


@dataclass(slots=True)
class RuntimeCourtroomResponse:
    """
    Response returned by a specific runtime (PRIMARY_CODER, DEEPSEEK_SYNTH, JUDGE)
    during live courtroom execution.
    """
    role: CourtroomRole
    content: str
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("Response content must not be blank.")

    @property
    def is_coder_response(self) -> bool:
        return self.role.is_coder

    @property
    def is_synth_response(self) -> bool:
        return self.role.is_synth

    @property
    def is_judge_response(self) -> bool:
        return self.role.is_judge

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        role: CourtroomRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeCourtroomResponse:
        return cls(
            role=role,
            content=content,
            metadata=metadata or {},
        )