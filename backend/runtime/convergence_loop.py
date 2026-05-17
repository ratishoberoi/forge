from __future__ import annotations
from dataclasses import dataclass
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.autonomous_courtroom import AutonomousCourtroom
from backend.runtime.revision_judge import RevisionJudge
from backend.runtime.revision_prompt import RevisionPromptBuilder


@dataclass(slots=True)
class ConvergenceResult:
    """
    Full result of a convergence loop run.
    Carries history, final verdict, and convergence metadata.
    """
    history: list[list[CognitionArtifact]]
    converged: bool
    iterations_run: int
    final_verdict: str
    final_objective: str

    @property
    def all_artifacts(self) -> list[CognitionArtifact]:
        return [a for round_artifacts in self.history for a in round_artifacts]

    @property
    def final_round(self) -> list[CognitionArtifact]:
        return self.history[-1] if self.history else []

    @property
    def coder_artifacts(self) -> list[CognitionArtifact]:
        return [a for a in self.all_artifacts if a.role == "PRIMARY_CODER"]

    @property
    def synth_artifacts(self) -> list[CognitionArtifact]:
        return [a for a in self.all_artifacts if a.role == "DEEPSEEK_SYNTH"]

    @property
    def judge_artifacts(self) -> list[CognitionArtifact]:
        return [a for a in self.all_artifacts if a.role == "JUDGE"]


class ConvergenceLoopError(Exception):
    """Raised when convergence loop encounters an unrecoverable error."""


class ConvergenceLoop:
    """
    Iterative autonomous refinement loop.
    Responsibilities:
    - execute courtroom rounds sequentially
    - ask RevisionJudge after each round
    - build refined objective via RevisionPromptBuilder
    - stop on convergence or max iterations
    - return full ConvergenceResult
    """

    SYNTH_INDEX = 1
    CODER_INDEX = 0
    JUDGE_INDEX = 2

    def __init__(
        self,
        *,
        courtroom: AutonomousCourtroom,
        judge: RevisionJudge,
        prompt_builder: RevisionPromptBuilder,
    ) -> None:
        self.courtroom = courtroom
        self.judge = judge
        self.prompt_builder = prompt_builder

    def run(
        self,
        *,
        objective: str,
        max_iterations: int = 3,
    ) -> list[list[CognitionArtifact]]:
        """
        Run convergence loop and return history.
        Backwards-compatible return type.
        For full result use run_full().
        """
        return self.run_full(
            objective=objective,
            max_iterations=max_iterations,
        ).history

    def run_full(
        self,
        *,
        objective: str,
        max_iterations: int = 3,
    ) -> ConvergenceResult:
        """
        Run convergence loop and return full ConvergenceResult.
        """
        if not objective.strip():
            raise ConvergenceLoopError("objective must not be blank.")
        if max_iterations < 1:
            raise ConvergenceLoopError(
                f"max_iterations must be >= 1, got {max_iterations}."
            )

        history: list[list[CognitionArtifact]] = []
        current_objective = objective
        final_verdict = ""
        converged = False

        for iteration in range(1, max_iterations + 1):
            artifacts = self.courtroom.execute(
                objective=current_objective,
                round_id=iteration,
            )
            history.append(artifacts)

            synth_artifact = artifacts[self.SYNTH_INDEX]

            final_verdict = self.judge.verdict(
                critique=synth_artifact.content,
                iteration=iteration,
                max_iterations=max_iterations,
            )

            should_continue = self.judge.should_continue(
                critique=synth_artifact.content,
                iteration=iteration,
                max_iterations=max_iterations,
            )

            if not should_continue:
                converged = not final_verdict.startswith("STOP — max")
                break

            current_objective = self.prompt_builder.build(
                objective=objective,
                coder_artifact=artifacts[self.CODER_INDEX],
                synth_artifact=synth_artifact,
            )

        return ConvergenceResult(
            history=history,
            converged=converged,
            iterations_run=len(history),
            final_verdict=final_verdict,
            final_objective=current_objective,
        )