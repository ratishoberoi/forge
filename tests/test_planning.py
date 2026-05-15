from backend.runtime.planning import AdaptiveCognitionPlanner
from backend.runtime.planning_policy import PlanningPolicy
from backend.runtime.planning_state import PlanningState


def build_state() -> PlanningState:
    return PlanningState(
        round_count=1,
        execution_failures=0,
        stagnation_count=0,
        architecture_risk=0.1,
        judge_confidence=0.2,
        context_pressure=0.1,
    )


def test_terminate_on_max_rounds():
    planner = AdaptiveCognitionPlanner(PlanningPolicy(max_rounds=2))
    state = build_state()
    state.round_count = 2
    decision = planner.decide(state)
    assert decision.terminate
    assert decision.reason == "max_rounds"


def test_terminate_on_stagnation():
    planner = AdaptiveCognitionPlanner(PlanningPolicy(stagnation_threshold=2))
    state = build_state()
    state.stagnation_count = 2
    decision = planner.decide(state)
    assert decision.terminate
    assert decision.reason == "stagnation"


def test_terminate_on_high_confidence():
    planner = AdaptiveCognitionPlanner(PlanningPolicy(confidence_threshold=0.8))
    state = build_state()
    state.judge_confidence = 0.95
    decision = planner.decide(state)
    assert decision.terminate
    assert decision.reason == "high_confidence"


def test_retry_invocation():
    planner = AdaptiveCognitionPlanner(PlanningPolicy(retry_failure_threshold=2))
    state = build_state()
    state.execution_failures = 2
    decision = planner.decide(state)
    assert decision.invoke_retry
    assert not decision.terminate


def test_architect_invocation():
    planner = AdaptiveCognitionPlanner(PlanningPolicy(architecture_risk_threshold=0.5))
    state = build_state()
    state.architecture_risk = 0.9
    decision = planner.decide(state)
    assert decision.invoke_architect
    assert not decision.terminate


def test_context_compression():
    planner = AdaptiveCognitionPlanner(PlanningPolicy(context_pressure_threshold=0.5))
    state = build_state()
    state.context_pressure = 0.9
    decision = planner.decide(state)
    assert decision.compress_context
    assert not decision.terminate


def test_continue_reason_on_normal_state():
    planner = AdaptiveCognitionPlanner(PlanningPolicy())
    decision = planner.decide(build_state())
    assert not decision.terminate
    assert decision.reason == "continue"


def test_multiple_flags_simultaneously():
    """High failures + high architecture risk should set both flags."""
    planner = AdaptiveCognitionPlanner(PlanningPolicy(
        retry_failure_threshold=2,
        architecture_risk_threshold=0.5,
    ))
    state = build_state()
    state.execution_failures = 2
    state.architecture_risk = 0.9
    decision = planner.decide(state)
    assert decision.invoke_retry
    assert decision.invoke_architect
    assert not decision.terminate