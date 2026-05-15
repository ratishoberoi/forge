from __future__ import annotations

from backend.runtime.courtroom_roles import CourtroomRole
from backend.runtime.courtroom_runtime import RuntimeCourtroomResponse


class CourtroomDispatcher:
    """
    Responsible for dispatching requests to the appropriate runtime
    and receiving structured responses.
    """

    def dispatch(
        self,
        *,
        role: CourtroomRole,
        content: str,          # This will later become prompt/context
        **metadata,
    ) -> RuntimeCourtroomResponse:
        """Dispatch to a runtime. Currently mocked — will connect to real LLM runtimes."""
        return RuntimeCourtroomResponse.create(
            role=role,
            content=content,
            metadata=metadata,
        )