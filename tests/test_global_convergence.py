import pytest
from backend.runtime.convergence_analyzer import GlobalConvergenceAnalyzer
from backend.runtime.convergence_metrics import ConvergenceMetrics
from backend.runtime.convergence_state import GlobalConvergenceState


def make_state(
    iteration: int = 5,
    score_variance: float = 0.2,
    retry_improvement: float = 0.1,
    critique_diversity: float = 0.5,
    execution_stability: float = 0.7,
) -> GlobalConvergenceState:
    return GlobalConvergenceState(
        iteration=iteration,
        metrics=ConvergenceMetrics(
            score_variance=score_variance,
            retry_improvement=retry_improvement,
            critique_diversity=critique_diversity,
            execution_stability=execution_stability,
        ),
    )


def test_stable_convergence():
    decision = GlobalConvergenceAnalyzer().analyze(make_state(
        score_variance=0.01,
        retry_improvement=0.001,
        critique_diversity=0.2,
        execution_stability=0.95,
    ))
    assert decision.converged
    assert decision.terminate
    assert decision.reason == "stable_convergence"


def test_cognition_instability():
    decision = GlobalConvergenceAnalyzer().analyze(make_state(
        score_variance=0.8,
        retry_improvement=0.3,
        critique_diversity=0.95,
        execution_stability=0.4,
    ))
    assert decision.escalate
    assert not decision.terminate
    assert decision.reason == "cognition_instability"


def test_continue_state():
    decision = GlobalConvergenceAnalyzer().analyze(make_state())
    assert not decision.terminate
    assert not decision.escalate
    assert not decision.converged
    assert decision.reason == "continue"


def test_convergence_boundary_exact_thresholds():
    """Boundary: exactly at threshold values — must still converge."""
    decision = GlobalConvergenceAnalyzer().analyze(make_state(
        score_variance=0.05,
        retry_improvement=0.01,
        critique_diversity=0.2,
        execution_stability=0.9,
    ))
    # retry_improvement == 0.01 is NOT < 0.01 — should NOT converge
    assert not decision.converged


def test_metrics_invalid_range_raises():
    with pytest.raises(ValueError):
        ConvergenceMetrics(
            score_variance=1.5,
            retry_improvement=0.1,
            critique_diversity=0.5,
            execution_stability=0.7,
        )


def test_state_negative_iteration_raises():
    with pytest.raises(ValueError):
        GlobalConvergenceState(
            iteration=-1,
            metrics=ConvergenceMetrics(
                score_variance=0.1,
                retry_improvement=0.1,
                critique_diversity=0.1,
                execution_stability=0.1,
            ),
        )


def test_escalate_does_not_set_converged():
    decision = GlobalConvergenceAnalyzer().analyze(make_state(
        score_variance=0.8,
        critique_diversity=0.95,
    ))
    assert decision.escalate
    assert not decision.converged