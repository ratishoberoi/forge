from __future__ import annotations
from backend.runtime.priority_policy import PriorityPolicy
from backend.runtime.priority_score import PriorityScore
from backend.runtime.priority_signal import PrioritySignal


class CognitionPriorityEngine:
    """
    Cognition attention allocation engine.
    Responsibilities:
    - compute weighted priority score from signals
    - classify into critical/high/normal tiers
    - drive strategic cognition budget allocation
    """

    def __init__(self, policy: PriorityPolicy) -> None:
        self.policy = policy

    def evaluate(self, signal: PrioritySignal) -> PriorityScore:
        score = round(
            signal.execution_risk * self.policy.execution_weight
            + signal.architecture_risk * self.policy.architecture_weight
            + signal.retry_pressure * self.policy.retry_weight
            + signal.convergence_instability * self.policy.instability_weight,
            6,
        )

        if score > 0.8:
            reason = "critical"
        elif score > 0.5:
            reason = "high"
        else:
            reason = "normal"

        return PriorityScore(score=score, reason=reason)