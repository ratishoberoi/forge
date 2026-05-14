from __future__ import annotations

from backend.runtime.convergence import ConvergenceTracker


def test_convergence_tracker_detects_repeated_scores() -> None:
    tracker = ConvergenceTracker()
    tracker.record(7.0, min_improvement=0.25)
    tracker.record(7.0, min_improvement=0.25)

    converged, reason = tracker.convergence_status(
        min_improvement=0.25,
        max_stagnant_rounds=2,
    )

    assert converged is True
    assert reason == "repeated_score_detected"


def test_convergence_tracker_detects_stagnation() -> None:
    tracker = ConvergenceTracker()
    tracker.record(6.0, min_improvement=0.5)
    tracker.record(6.1, min_improvement=0.5)
    tracker.record(6.15, min_improvement=0.5)

    converged, reason = tracker.convergence_status(
        min_improvement=0.5,
        max_stagnant_rounds=2,
    )

    assert converged is True
    assert reason == "stagnation_detected"


def test_convergence_tracker_detects_oscillation() -> None:
    tracker = ConvergenceTracker()
    tracker.record(6.0, min_improvement=0.1)
    tracker.record(7.0, min_improvement=0.1)
    tracker.record(6.0, min_improvement=0.1)
    tracker.record(7.0, min_improvement=0.1)

    converged, reason = tracker.convergence_status(
        min_improvement=0.1,
        max_stagnant_rounds=3,
    )

    assert converged is True
    assert reason == "oscillation_detected"
