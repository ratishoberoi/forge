import pytest
from backend.runtime.recovery_manager import RecoveryManager
from backend.runtime.recovery_policy import RecoveryPolicy
from backend.runtime.recovery_state import RecoveryState


def build_state(iteration: int = 2) -> RecoveryState:
    return RecoveryState(
        iteration=iteration,
        active_role="PRIMARY_CODER",
        last_completed_node="inspect_routes",
        replay_offset=5,
    )


def test_checkpoint_creation():
    manager = RecoveryManager(RecoveryPolicy())
    checkpoint = manager.create_checkpoint(build_state())
    assert checkpoint.state.iteration == 2
    assert checkpoint.state.active_role == "PRIMARY_CODER"
    assert checkpoint.checkpoint_id


def test_checkpoint_id_is_unique():
    manager = RecoveryManager(RecoveryPolicy())
    a = manager.create_checkpoint(build_state())
    b = manager.create_checkpoint(build_state())
    assert a.checkpoint_id != b.checkpoint_id


def test_recovery_attempt_limit():
    manager = RecoveryManager(RecoveryPolicy(max_recovery_attempts=2))
    assert manager.can_recover()
    manager.mark_recovery_attempt()
    assert manager.can_recover()
    manager.mark_recovery_attempt()
    assert not manager.can_recover()


def test_attempts_remaining():
    manager = RecoveryManager(RecoveryPolicy(max_recovery_attempts=3))
    assert manager.attempts_remaining == 3
    manager.mark_recovery_attempt()
    assert manager.attempts_remaining == 2


def test_latest_checkpoint_none_initially():
    manager = RecoveryManager(RecoveryPolicy())
    assert manager.latest_checkpoint() is None


def test_latest_checkpoint_returns_last():
    manager = RecoveryManager(RecoveryPolicy())
    manager.create_checkpoint(build_state(iteration=1))
    manager.create_checkpoint(build_state(iteration=5))
    assert manager.latest_checkpoint().state.iteration == 5


def test_should_checkpoint_every_iteration():
    manager = RecoveryManager(RecoveryPolicy(checkpoint_every_n_iterations=1))
    assert manager.should_checkpoint(0)
    assert manager.should_checkpoint(3)


def test_should_checkpoint_every_n():
    manager = RecoveryManager(RecoveryPolicy(checkpoint_every_n_iterations=3))
    assert manager.should_checkpoint(0)
    assert manager.should_checkpoint(3)
    assert not manager.should_checkpoint(1)
    assert not manager.should_checkpoint(2)


def test_recovery_state_invalid_iteration_raises():
    with pytest.raises(ValueError):
        RecoveryState(
            iteration=-1,
            active_role="PRIMARY_CODER",
            last_completed_node="node",
            replay_offset=0,
        )


def test_recovery_state_empty_role_raises():
    with pytest.raises(ValueError):
        RecoveryState(
            iteration=0,
            active_role="",
            last_completed_node="node",
            replay_offset=0,
        )


def test_recovery_policy_invalid_attempts_raises():
    with pytest.raises(ValueError):
        RecoveryPolicy(max_recovery_attempts=0)