"""Structured patch judging and candidate comparison."""

from __future__ import annotations

from dataclasses import dataclass

from backend.runtime.candidate import CandidateCollection, PatchCandidate
from backend.runtime.patches import PatchRisk


@dataclass(slots=True)
class JudgeScore:
    correctness: float
    architecture: float
    safety: float
    minimality: float
    maintainability: float
    hallucination_risk: float

    @property
    def composite_score(self) -> float:
        positive_total = (
            self.correctness
            + self.architecture
            + self.safety
            + self.minimality
            + self.maintainability
            + (10.0 - self.hallucination_risk)
        )
        return round(positive_total / 6.0, 3)


@dataclass(slots=True)
class JudgeResult:
    candidate_id: str
    score: JudgeScore
    reasoning: str
    critique_summary: str
    recommendation: str
    retry_recommended: bool
    winning_candidate_id: str | None = None


class PatchJudge:
    """Deterministic patch evaluator for candidate ranking and critique generation."""

    def evaluate_candidate(self, candidate: PatchCandidate) -> JudgeResult:
        patch = candidate.patch
        diff_lines = patch.unified_diff.splitlines()
        impacted_files = len(patch.impacted_files)
        validation_penalty = min(len(patch.validation_errors) * 2.0, 6.0)
        risk_penalty = {
            PatchRisk.LOW: 0.0,
            PatchRisk.MEDIUM: 1.5,
            PatchRisk.HIGH: 3.0,
            PatchRisk.UNKNOWN: 2.0,
        }[patch.risk]

        correctness = max(0.0, 9.0 - validation_penalty)
        architecture = max(0.0, 9.0 - max(0, impacted_files - 2) * 0.75)
        safety = max(0.0, 9.0 - risk_penalty - validation_penalty / 2.0)
        minimality = max(0.0, 10.0 - min(len(diff_lines) / 20.0, 8.0))
        maintainability = 8.5 if patch.reasoning else 7.0
        hallucination_risk = min(10.0, 2.0 + validation_penalty + (0.5 if not patch.summary else 0.0))

        score = JudgeScore(
            correctness=round(correctness, 3),
            architecture=round(architecture, 3),
            safety=round(safety, 3),
            minimality=round(minimality, 3),
            maintainability=round(maintainability, 3),
            hallucination_risk=round(hallucination_risk, 3),
        )

        recommendation = self._recommend(score)
        reasoning = self._build_reasoning(candidate, score)
        critique = self._build_critique(candidate, score)
        return JudgeResult(
            candidate_id=candidate.candidate_id,
            score=score,
            reasoning=reasoning,
            critique_summary=critique,
            recommendation=recommendation,
            retry_recommended=recommendation != "accept",
        )

    def evaluate_candidates(self, candidates: CandidateCollection | list[PatchCandidate]) -> list[JudgeResult]:
        candidate_list = candidates.all() if isinstance(candidates, CandidateCollection) else list(candidates)
        results = [self.evaluate_candidate(candidate) for candidate in candidate_list]
        ranked = sorted(
            results,
            key=lambda result: (-result.score.composite_score, result.candidate_id),
        )
        winning_candidate_id = ranked[0].candidate_id if ranked else None
        for result in ranked:
            result.winning_candidate_id = winning_candidate_id
        return ranked

    def select_best_patch(self, candidates: CandidateCollection | list[PatchCandidate]) -> JudgeResult | None:
        results = self.evaluate_candidates(candidates)
        return results[0] if results else None

    def critique_summaries(self, candidates: CandidateCollection | list[PatchCandidate]) -> dict[str, str]:
        return {
            result.candidate_id: result.critique_summary
            for result in self.evaluate_candidates(candidates)
        }

    @staticmethod
    def _recommend(score: JudgeScore) -> str:
        if score.composite_score >= 8.0 and score.hallucination_risk <= 3.0:
            return "accept"
        if score.composite_score >= 6.0:
            return "revise"
        return "reject"

    @staticmethod
    def _build_reasoning(candidate: PatchCandidate, score: JudgeScore) -> str:
        patch = candidate.patch
        return (
            f"Candidate {candidate.candidate_id} touches {len(patch.impacted_files)} file(s) with "
            f"composite score {score.composite_score}. Safety={score.safety}, "
            f"architecture={score.architecture}, minimality={score.minimality}."
        )

    @staticmethod
    def _build_critique(candidate: PatchCandidate, score: JudgeScore) -> str:
        patch = candidate.patch
        validation_note = (
            f"Validation issues: {', '.join(patch.validation_errors)}."
            if patch.validation_errors
            else "No validator failures recorded."
        )
        return (
            f"Patch '{patch.title}' scored {score.composite_score}. "
            f"{validation_note} Hallucination risk={score.hallucination_risk}."
        )
