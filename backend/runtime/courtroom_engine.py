from __future__ import annotations

from backend.runtime.courtroom_dispatcher import CourtroomDispatcher
from backend.runtime.courtroom_roles import CourtroomRole
from backend.runtime.courtroom_session import CourtroomSession


class CourtroomEngine:
    """
    Core live execution engine for courtroom collaborative cognition.
    Orchestrates the full PrimaryCoder → DeepSeekSynth → Judge flow.
    """

    def __init__(self, dispatcher: CourtroomDispatcher) -> None:
        self.dispatcher = dispatcher

    def execute(self, *, objective: str, **session_metadata) -> CourtroomSession:
        """
        Execute a full courtroom cognition round.
        This is where real multi-runtime collaboration will happen.
        """
        session = CourtroomSession(
            objective=objective,
            metadata=session_metadata,
        )

        # Step 1: Primary Coder generates initial patch
        coder_response = self.dispatcher.dispatch(
            role=CourtroomRole.PRIMARY_CODER,
            content=f"Generate implementation for: {objective}",
        )
        session.add_response(coder_response)

        # Step 2: DeepSeekSynth provides architecture critique
        synth_response = self.dispatcher.dispatch(
            role=CourtroomRole.DEEPSEEK_SYNTH,
            content=f"Review the following implementation for risks and improvements: {coder_response.content}",
        )
        session.add_response(synth_response)

        # Step 3: Judge gives final convergence verdict
        judge_response = self.dispatcher.dispatch(
            role=CourtroomRole.JUDGE,
            content=f"Evaluate convergence and safety of the revised solution for: {objective}",
        )
        session.add_response(judge_response)

        return session