from __future__ import annotations
from backend.runtime.artifact import CognitionArtifact


class RevisionPromptBuilder:
    """
    Builds iterative revision prompts using prior cognition artifacts.
    Responsibilities:
    - inject prior patch and critique into next prompt
    - support optional judge verdict context
    - support multi-round history injection
    - keep prompt within char budget
    """

    DEFAULT_MAX_CHARS = 8_000

    def build(
        self,
        *,
        objective: str,
        coder_artifact: CognitionArtifact,
        synth_artifact: CognitionArtifact,
        judge_artifact: CognitionArtifact | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """
        Build revision prompt from prior round artifacts.
        Optionally includes judge verdict for richer context.
        """
        if not objective.strip():
            raise ValueError("objective must not be blank.")

        parts = [
            f"OBJECTIVE:\n{objective}",
            f"PREVIOUS PATCH:\n{coder_artifact.content[:max_chars]}",
            f"ARCHITECTURE CRITIQUE:\n{synth_artifact.content[:max_chars]}",
        ]

        if judge_artifact is not None:
            parts.append(
                f"JUDGE VERDICT:\n{judge_artifact.content[:max_chars]}"
            )

        parts.append(
            "Revise the implementation to address the critique "
            "while preserving the objective."
        )

        return "\n\n".join(parts)

    def build_from_history(
        self,
        *,
        objective: str,
        history: list[list[CognitionArtifact]],
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """
        Build revision prompt from full round history.
        Uses the most recent round's artifacts.
        Raises ValueError if history is empty.
        """
        if not history:
            raise ValueError("history must not be empty.")

        latest_round = history[-1]

        if len(latest_round) < 2:
            raise ValueError(
                f"Latest round must have at least 2 artifacts "
                f"(coder + synth), got {len(latest_round)}."
            )

        coder = latest_round[0]
        synth = latest_round[1]
        judge = latest_round[2] if len(latest_round) > 2 else None

        return self.build(
            objective=objective,
            coder_artifact=coder,
            synth_artifact=synth,
            judge_artifact=judge,
            max_chars=max_chars,
        )

    def build_multi_round_context(
        self,
        *,
        objective: str,
        history: list[list[CognitionArtifact]],
        max_rounds: int = 3,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """
        Build context from last N rounds of history.
        Useful when judge needs full evolution visible.
        """
        if not history:
            raise ValueError("history must not be empty.")

        recent = history[-max_rounds:]
        sections = [f"OBJECTIVE:\n{objective}"]

        for i, round_artifacts in enumerate(recent, start=1):
            round_num = len(history) - len(recent) + i
            coder_content = round_artifacts[0].content[:max_chars] if round_artifacts else ""
            synth_content = round_artifacts[1].content[:max_chars] if len(round_artifacts) > 1 else ""
            sections.append(
                f"--- ROUND {round_num} ---\n"
                f"PATCH:\n{coder_content}\n\n"
                f"CRITIQUE:\n{synth_content}"
            )

        sections.append(
            "Revise the implementation to address all critiques "
            "while preserving the objective."
        )

        return "\n\n".join(sections)