import pytest
from backend.runtime.priority_engine import CognitionPriorityEngine
from backend.runtime.priority_policy import PriorityPolicy
from backend.runtime.priority_signal import PrioritySignal


def make_engine() -> CognitionPriorityEngine:
    return CognitionPriorityEngine(PriorityPolicy())


def make_signal(
    execution_risk: float = 0.1,
    architecture_risk: float = 0.1,
    retry_pressure: float = 0.1,
    convergence_instability: float = 0.1,
) -> PrioritySignal:
    return PrioritySignal(
        execution_risk=execution_risk,
        architecture_risk=architecture_risk,
        retry_pressure=retry_pressure,
        convergence_instability=convergence_instability,
    )


def test_critical_priority():
    result = make_engine().evaluate(make_signal(1.0, 1.0, 1.0, 1.0))
    assert result.reason == "critical"
    assert result.score > 0.8
    assert result.is_critical


def test_high_priority():
    result = make_engine().evaluate(make_signal(
        execution_risk=0.7,
        architecture_risk=0.5,
        retry_pressure=0.4,
        convergence_instability=0.2,
    ))
    assert result.reason == "high"
    assert result.is_high


def test_normal_priority():
    result = make_engine().evaluate(make_signal(0.1, 0.1, 0.1, 0.1))
    assert result.reason == "normal"
    assert result.is_normal


def test_score_is_weighted_sum():
    """Verify score matches manual weighted calculation."""
    policy = PriorityPolicy(
        execution_weight=0.4,
        architecture_weight=0.3,
        retry_weight=0.2,
        instability_weight=0.1,
    )
    signal = make_signal(0.5, 0.5, 0.5, 0.5)
    result = CognitionPriorityEngine(policy).evaluate(signal)
    expected = 0.5 * 0.4 + 0.5 * 0.3 + 0.5 * 0.2 + 0.5 * 0.1
    assert abs(result.score - expected) < 1e-6


def test_zero_signal_is_normal():
    result = make_engine().evaluate(make_signal(0.0, 0.0, 0.0, 0.0))
    assert result.reason == "normal"
    assert result.score == 0.0


def test_custom_weights_shift_priority():
    """High execution weight makes execution_risk dominate."""
    policy = PriorityPolicy(
        execution_weight=0.97,
        architecture_weight=0.01,
        retry_weight=0.01,
        instability_weight=0.01,
    )
    result = CognitionPriorityEngine(policy).evaluate(
        make_signal(execution_risk=1.0, architecture_risk=0.0, retry_pressure=0.0, convergence_instability=0.0)
    )
    assert result.is_critical


def test_signal_invalid_range_raises():
    with pytest.raises(ValueError):
        make_signal(execution_risk=1.5)


def test_policy_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        PriorityPolicy(
            execution_weight=0.5,
            architecture_weight=0.5,
            retry_weight=0.5,
            instability_weight=0.5,
        )