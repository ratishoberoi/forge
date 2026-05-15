from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

from backend.runtime.courtroom_roles import CourtroomRole          
from backend.runtime.courtroom_runtime import RuntimeCourtroomResponse


@dataclass(slots=True)
class CourtroomSession:
    """
    Live execution session of a courtroom cognition round.
    Tracks the full conversation across multiple runtimes.
    """
    objective: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    responses: list[RuntimeCourtroomResponse] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective or not self.objective.strip():
            raise ValueError("Session objective must not be blank.")

    def add_response(self, response: RuntimeCourtroomResponse) -> None:
        self.responses.append(response)

    def add_coder_response(self, content: str, **metadata) -> RuntimeCourtroomResponse:
        resp = RuntimeCourtroomResponse.create(
            role=CourtroomRole.PRIMARY_CODER, content=content, metadata=metadata
        )
        self.add_response(resp)
        return resp

    def add_synth_response(self, content: str, **metadata) -> RuntimeCourtroomResponse:
        resp = RuntimeCourtroomResponse.create(
            role=CourtroomRole.DEEPSEEK_SYNTH, content=content, metadata=metadata
        )
        self.add_response(resp)
        return resp

    def add_judge_response(self, content: str, **metadata) -> RuntimeCourtroomResponse:
        resp = RuntimeCourtroomResponse.create(
            role=CourtroomRole.JUDGE, content=content, metadata=metadata
        )
        self.add_response(resp)
        return resp

    @property
    def response_count(self) -> int:
        return len(self.responses)

    @property
    def last_response(self) -> RuntimeCourtroomResponse | None:
        return self.responses[-1] if self.responses else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "objective": self.objective,
            "responses": [r.to_dict() for r in self.responses],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }