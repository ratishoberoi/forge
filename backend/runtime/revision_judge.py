from __future__ import annotations


class RevisionJudge:
    """
    Determines whether autonomous refinement should continue.
    Uses convergence scoring instead of binary signal detection.
    Score closer to 1.0 = stable. Score closer to 0.0 = unstable.
    """

    SEVERE_TOKENS = ("unsafe", "critical", "failure", "broken")
    MODERATE_TOKENS = ("risk", "issue", "bug", "incorrect")

    SEVERE_PENALTY = 0.4
    MODERATE_PENALTY = 0.2

    def convergence_score(self, *, critique: str) -> float:
        """
        Score critique stability from 0.0 (unstable) to 1.0 (stable).
        Each severe token subtracts 0.4, each moderate token subtracts 0.2.
        Clamped to [0.0, 1.0].
        """
        critique_lower = critique.lower()
        score = 1.0

        for token in self.SEVERE_TOKENS:
            if token in critique_lower:
                score -= self.SEVERE_PENALTY

        for token in self.MODERATE_TOKENS:
            if token in critique_lower:
                score -= self.MODERATE_PENALTY

        return max(score, 0.0)

    def should_continue(
        self,
        *,
        critique: str,
        iteration: int,
        max_iterations: int,
        threshold: float = 0.75,
    ) -> bool:
        """
        Return True if refinement should continue.
        Continues when score < threshold and iteration < max_iterations.
        """
        if iteration >= max_iterations:
            return False

        score = self.convergence_score(critique=critique)
        return score < threshold

    def verdict(
        self,
        *,
        critique: str,
        iteration: int,
        max_iterations: int,
        threshold: float = 0.75,
    ) -> str:
        """
        Return human-readable verdict string with score.
        Useful for logging and audit trails.
        """
        if iteration >= max_iterations:
            return f"STOP — max iterations ({max_iterations}) reached."

        score = self.convergence_score(critique=critique)

        if score >= threshold:
            return f"STOP — convergence score {score:.2f} >= threshold {threshold:.2f}."

        return f"CONTINUE — convergence score {score:.2f} < threshold {threshold:.2f}."

    def score_breakdown(self, critique: str) -> dict:
        """
        Return per-token score breakdown.
        Useful for debugging convergence behavior.
        """
        critique_lower = critique.lower()
        matched_severe = [t for t in self.SEVERE_TOKENS if t in critique_lower]
        matched_moderate = [t for t in self.MODERATE_TOKENS if t in critique_lower]
        score = self.convergence_score(critique=critique)

        return {
            "score": score,
            "matched_severe": matched_severe,
            "matched_moderate": matched_moderate,
            "severe_penalty": len(matched_severe) * self.SEVERE_PENALTY,
            "moderate_penalty": len(matched_moderate) * self.MODERATE_PENALTY,
        }