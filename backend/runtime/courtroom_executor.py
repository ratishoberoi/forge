from __future__ import annotations

from backend.runtime.courtroom_message import CourtroomMessage
from backend.runtime.courtroom_pipeline import CourtroomPipeline
from backend.runtime.courtroom_roles import CourtroomRole


class CourtroomExecutor:
    """
    High-level orchestrator for building and executing courtroom cognition flows.
    """

    def create_pipeline(self, **metadata) -> CourtroomPipeline:
        """Create a fresh cognition pipeline."""
        return CourtroomPipeline(metadata=metadata)

    def coder_message(self, content: str, **metadata) -> CourtroomMessage:
        return CourtroomMessage.create(
            role=CourtroomRole.PRIMARY_CODER, content=content, metadata=metadata
        )

    def synth_message(self, content: str, **metadata) -> CourtroomMessage:
        return CourtroomMessage.create(
            role=CourtroomRole.DEEPSEEK_SYNTH, content=content, metadata=metadata
        )

    def judge_message(self, content: str, **metadata) -> CourtroomMessage:
        return CourtroomMessage.create(
            role=CourtroomRole.JUDGE, content=content, metadata=metadata
        )

    # === Convenience methods that operate on pipeline directly ===
    def add_coder_message(
        self, pipeline: CourtroomPipeline, content: str, **metadata
    ) -> CourtroomMessage:
        msg = self.coder_message(content, **metadata)
        self.append(pipeline=pipeline, message=msg)
        return msg

    def add_synth_message(
        self, pipeline: CourtroomPipeline, content: str, **metadata
    ) -> CourtroomMessage:
        msg = self.synth_message(content, **metadata)
        self.append(pipeline=pipeline, message=msg)
        return msg

    def add_judge_message(
        self, pipeline: CourtroomPipeline, content: str, **metadata
    ) -> CourtroomMessage:
        msg = self.judge_message(content, **metadata)
        self.append(pipeline=pipeline, message=msg)
        return msg

    def append(
        self, *, pipeline: CourtroomPipeline, message: CourtroomMessage
    ) -> None:
        """Safely append message to pipeline."""
        pipeline.add_message(message)

    def run_full_round(
        self,
        pipeline: CourtroomPipeline,
        coder_patch: str,
        synth_critique: str,
        coder_revision: str,
        judge_verdict: str,
    ) -> None:
        """Convenience method for the classic PrimaryCoder → Synth → Coder → Judge flow."""
        self.add_coder_message(pipeline, coder_patch)
        self.add_synth_message(pipeline, synth_critique)
        self.add_coder_message(pipeline, coder_revision)
        self.add_judge_message(pipeline, judge_verdict)