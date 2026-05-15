from __future__ import annotations
from backend.runtime.convergence_decision import ConvergenceDecision
from backend.runtime.convergence_state import GlobalConvergenceState


class GlobalConvergenceAnalyzer:
    """
    Evaluates whether autonomous cognition is stabilizing.
    Prevents:
    - infinite retry spirals
    - oscillating cognition
    - degenerate loops
    """

    def analyze(self, state: GlobalConvergenceState) -> ConvergenceDecision:
        m = state.metrics

        if (
            m.retry_improvement < 0.01
            and m.score_variance < 0.05
            and m.execution_stability > 0.9
        ):
            return ConvergenceDecision(
                converged=True,
                terminate=True,
                reason="stable_convergence",
            )

        if m.critique_diversity > 0.9 and m.score_variance > 0.5:
            return ConvergenceDecision(
                escalate=True,
                reason="cognition_instability",
            )

        return ConvergenceDecision(reason="continue")