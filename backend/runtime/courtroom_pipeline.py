from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid
from backend.runtime.courtroom_message import CourtroomMessage
from backend.runtime.courtroom_roles import CourtroomRole


@dataclass(slots=True)
class CourtroomPipeline:
    """
    Sequential chain of courtroom messages representing a full cognition flow.
    """
    messages: list[CourtroomMessage] = field(default_factory=list)
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # noqa: F821
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: CourtroomMessage) -> None:
        """Append a message to the pipeline."""
        self.messages.append(message)

    def add_coder_message(self, content: str, **metadata) -> CourtroomMessage:
        msg = CourtroomMessage.create(
            role=CourtroomRole.PRIMARY_CODER,
            content=content,
            metadata=metadata,
        )
        self.add_message(msg)
        return msg

    def add_synth_message(self, content: str, **metadata) -> CourtroomMessage:
        msg = CourtroomMessage.create(
            role=CourtroomRole.DEEPSEEK_SYNTH,
            content=content,
            metadata=metadata,
        )
        self.add_message(msg)
        return msg

    def add_judge_message(self, content: str, **metadata) -> CourtroomMessage:
        msg = CourtroomMessage.create(
            role=CourtroomRole.JUDGE,
            content=content,
            metadata=metadata,
        )
        self.add_message(msg)
        return msg

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_message(self) -> CourtroomMessage | None:
        return self.messages[-1] if self.messages else None

    def get_messages_by_role(self, role: CourtroomRole) -> list[CourtroomMessage]:
        return [m for m in self.messages if m.role == role]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }