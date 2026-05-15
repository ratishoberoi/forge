from __future__ import annotations
from backend.runtime.planning_decision import PlanningDecision
from backend.runtime.planning_policy import PlanningPolicy
from backend.runtime.planning_state import PlanningState


class AdaptiveCognitionPlanner:
    """
    Adaptive cognition strategy engine.
    Decides per-round:
    - whether to terminate
    - whether to escalate to retry/architect
    - whether to compress context
    """

    def __init__(self, policy: PlanningPolicy) -> None:
        self.policy = policy

    def decide(self, state: PlanningState) -> PlanningDecision:
        if state.round_count >= self.policy.max_rounds:
            return PlanningDecision(terminate=True, reason="max_rounds")

        if state.stagnation_count >= self.policy.stagnation_threshold:
            return PlanningDecision(terminate=True, reason="stagnation")

        if state.judge_confidence >= self.policy.confidence_threshold:
            return PlanningDecision(terminate=True, reason="high_confidence")

        decision = PlanningDecision()

        if state.execution_failures >= self.policy.retry_failure_threshold:
            decision.invoke_retry = True

        if state.architecture_risk >= self.policy.architecture_risk_threshold:
            decision.invoke_architect = True

        if state.context_pressure >= self.policy.context_pressure_threshold:
            decision.compress_context = True

        decision.reason = "continue"
        return decision