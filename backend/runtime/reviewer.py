"""Independent candidate review for courtroom-style orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from backend.runtime.candidate import PatchCandidate
from backend.runtime.execution_review import ExecutionSeverity
from backend.runtime.judge import JudgeResult
from backend.runtime.patches import PatchRisk


@dataclass(slots=True)
class CandidateReview:
    candidate_id: str
    critique_summary: str
    weaknesses: tuple[str, ...]
    architecture_risk: str
    maintainability_score: float
    hallucination_risk: float
    execution_risk: float
    adversarial_question: str

    @property
    def review_score(self) -> float:
        positive_total = self.maintainability_score + (10.0 - self.hallucination_risk) + (10.0 - self.execution_risk)
        return round(positive_total / 3.0, 3)


class CandidateReviewer:
    """Produces deterministic independent critiques for patch candidates."""

    def review(
        self,
        candidate: PatchCandidate,
        judge_result: JudgeResult | None = None,
    ) -> CandidateReview:
        patch = candidate.patch
        weaknesses: list[str] = []

        if len(patch.impacted_files) > 2:
            weaknesses.append("Patch touches multiple files and may widen architectural impact.")
        if patch.risk in {PatchRisk.HIGH, PatchRisk.UNKNOWN}:
            weaknesses.append(f"Patch risk is marked {patch.risk.value}.")
        if patch.validation_errors:
            weaknesses.append(f"Validator reported: {', '.join(patch.validation_errors)}.")
        if not patch.reasoning:
            weaknesses.append("Patch reasoning is missing.")
        if not patch.summary:
            weaknesses.append("Patch summary is missing.")

        execution_risk = 2.0
        if judge_result is not None and judge_result.execution_review is not None:
            execution_review = judge_result.execution_review
            if execution_review.severity is ExecutionSeverity.INFO:
                execution_risk = 1.0
            elif execution_review.severity is ExecutionSeverity.WARNING:
                execution_risk = 5.5
                weaknesses.append(f"Execution warning: {execution_review.failure_summary}")
            else:
                execution_risk = 8.5
                weaknesses.append(f"Critical execution failure: {execution_review.failure_summary}")

        architecture_risk = self._architecture_risk(candidate)
        maintainability_score = self._maintainability_score(candidate, judge_result)
        hallucination_risk = self._hallucination_risk(candidate, judge_result)
        adversarial_question = self._adversarial_question(candidate, weaknesses)

        critique_summary = (
            f"Candidate {candidate.candidate_id} architecture risk is {architecture_risk}; "
            f"maintainability={maintainability_score}, hallucination_risk={hallucination_risk}, "
            f"execution_risk={round(execution_risk, 3)}."
        )
        if weaknesses:
            critique_summary = f"{critique_summary} Weaknesses: {' '.join(weaknesses)}"
        else:
            critique_summary = f"{critique_summary} No material weaknesses identified."

        return CandidateReview(
            candidate_id=candidate.candidate_id,
            critique_summary=critique_summary,
            weaknesses=tuple(weaknesses),
            architecture_risk=architecture_risk,
            maintainability_score=maintainability_score,
            hallucination_risk=hallucination_risk,
            execution_risk=round(execution_risk, 3),
            adversarial_question=adversarial_question,
        )

    @staticmethod
    def _architecture_risk(candidate: PatchCandidate) -> str:
        patch = candidate.patch
        if len(patch.impacted_files) > 3 or patch.risk is PatchRisk.HIGH:
            return "high"
        if len(patch.impacted_files) > 1 or patch.risk in {PatchRisk.MEDIUM, PatchRisk.UNKNOWN}:
            return "medium"
        return "low"

    @staticmethod
    def _maintainability_score(
        candidate: PatchCandidate,
        judge_result: JudgeResult | None,
    ) -> float:
        score = 8.5
        if not candidate.patch.reasoning:
            score -= 1.5
        if candidate.patch.validation_errors:
            score -= min(2.0, len(candidate.patch.validation_errors) * 0.75)
        if judge_result is not None:
            score = min(score, judge_result.score.maintainability)
        return round(max(0.0, score), 3)

    @staticmethod
    def _hallucination_risk(
        candidate: PatchCandidate,
        judge_result: JudgeResult | None,
    ) -> float:
        risk = 2.0
        if not candidate.patch.summary:
            risk += 1.0
        if candidate.patch.validation_errors:
            risk += min(4.0, len(candidate.patch.validation_errors) * 1.25)
        if judge_result is not None:
            risk = max(risk, judge_result.score.hallucination_risk)
        return round(min(10.0, risk), 3)

    @staticmethod
    def _adversarial_question(candidate: PatchCandidate, weaknesses: list[str]) -> str:
        if weaknesses:
            return (
                f"What evidence shows candidate {candidate.candidate_id} resolves "
                f"the identified weaknesses without expanding scope?"
            )
        return f"What proof demonstrates candidate {candidate.candidate_id} is minimal and architecture-safe?"
