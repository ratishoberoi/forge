from __future__ import annotations

from enum import Enum


class CourtroomRole(str, Enum):
    """
    Roles in the collaborative cognition courtroom.
    Defines who participates in the structured debate.
    """
    PRIMARY_CODER = "PRIMARY_CODER"
    DEEPSEEK_SYNTH = "DEEPSEEK_SYNTH"
    JUDGE = "JUDGE"

    @property
    def is_coder(self) -> bool:
        return self == CourtroomRole.PRIMARY_CODER

    @property
    def is_synth(self) -> bool:
        return self == CourtroomRole.DEEPSEEK_SYNTH

    @property
    def is_judge(self) -> bool:
        return self == CourtroomRole.JUDGE