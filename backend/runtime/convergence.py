"""Convergence tracking for bounded retry orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ConvergenceTracker:
    """Tracks score progression for retry convergence and stagnation detection."""

    score_history: list[float] = field(default_factory=list)
    stagnant_rounds: int = 0

    def record(self, score: float, *, min_improvement: float) -> None:
        if self.score_history:
            improvement = score - self.score_history[-1]
            if improvement < min_improvement:
                self.stagnant_rounds += 1
            else:
                self.stagnant_rounds = 0
        self.score_history.append(score)

    def detect_repeated_scores(self, *, tolerance: float = 0.0) -> bool:
        if len(self.score_history) < 2:
            return False
        latest = self.score_history[-1]
        previous = self.score_history[-2]
        return abs(latest - previous) <= tolerance

    def detect_non_improving_retries(self, *, min_improvement: float, max_stagnant_rounds: int) -> bool:
        if len(self.score_history) < 2:
            return False
        latest = self.score_history[-1]
        previous = self.score_history[-2]
        improvement = latest - previous
        return improvement < min_improvement and self.stagnant_rounds >= max_stagnant_rounds

    def detect_retry_oscillation(self, *, tolerance: float = 0.0) -> bool:
        if len(self.score_history) < 4:
            return False
        a, b, c, d = self.score_history[-4:]
        return abs(a - c) <= tolerance and abs(b - d) <= tolerance and abs(a - b) > tolerance

    def convergence_status(
        self,
        *,
        min_improvement: float,
        max_stagnant_rounds: int,
        tolerance: float = 0.0,
    ) -> tuple[bool, str | None]:
        if self.detect_retry_oscillation(tolerance=tolerance):
            return True, "oscillation_detected"
        if self.detect_non_improving_retries(
            min_improvement=min_improvement,
            max_stagnant_rounds=max_stagnant_rounds,
        ):
            return True, "stagnation_detected"
        if self.detect_repeated_scores(tolerance=tolerance):
            return True, "repeated_score_detected"
        return False, None
