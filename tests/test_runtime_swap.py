import pytest
from backend.runtime.runtime_process import RuntimeProcess
from backend.runtime.runtime_swap_engine import RuntimeSwapEngine


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeLauncher:
    """Records all launch() calls."""

    def __init__(self) -> None:
        self.launched: list[str] = []

    def launch(self, process: RuntimeProcess) -> RuntimeProcess:
        process.mark_launched(pid=999)
        self.launched.append(process.role)
        return process


class FakeShutdown:
    """Records all shutdown() calls."""

    def __init__(self) -> None:
        self.shutdowns: list[str] = []

    def shutdown(self, process: RuntimeProcess) -> None:
        process.mark_stopped()
        self.shutdowns.append(process.role)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_process(role: str, port: int = 8000) -> RuntimeProcess:
    return RuntimeProcess(role=role, model="fake-model", port=port)


def make_engine(
    launcher: FakeLauncher | None = None,
    shutdown: FakeShutdown | None = None,
) -> tuple[RuntimeSwapEngine, FakeLauncher, FakeShutdown]:
    launcher = launcher or FakeLauncher()
    shutdown = shutdown or FakeShutdown()
    engine = RuntimeSwapEngine(launcher=launcher, shutdown=shutdown)
    return engine, launcher, shutdown


# ── RuntimeProcess ────────────────────────────────────────────────────────────

def test_process_initial_state():
    process = build_process("coder")
    assert process.active is False
    assert process.pid is None
    assert process.is_running is False


def test_process_mark_launched():
    process = build_process("coder")
    process.mark_launched(pid=1234)
    assert process.pid == 1234
    assert process.active is True
    assert process.is_running is True
    assert process.launched_at is not None


def test_process_mark_stopped():
    process = build_process("coder")
    process.mark_launched(pid=1234)
    process.mark_stopped()
    assert process.active is False


def test_process_base_url():
    process = build_process("coder", port=8001)
    assert process.base_url == "http://127.0.0.1:8001"


def test_process_invalid_port_raises():
    with pytest.raises(ValueError, match="port"):
        RuntimeProcess(role="coder", model="fake-model", port=80)


def test_process_empty_role_raises():
    with pytest.raises(ValueError, match="role"):
        RuntimeProcess(role="", model="fake-model", port=8000)


def test_process_to_dict():
    process = build_process("coder")
    d = process.to_dict()
    assert d["role"] == "coder"
    assert d["active"] is False
    assert "launched_at" in d


# ── RuntimeSwapEngine ─────────────────────────────────────────────────────────

def test_runtime_swap_first_process_launched():
    engine, launcher, shutdown = make_engine()

    first = engine.swap(build_process("coder"))

    assert first.active is True
    assert first.pid == 999
    assert engine.active_role == "coder"
    assert shutdown.shutdowns == []


def test_runtime_swap_second_shuts_down_first():
    engine, launcher, shutdown = make_engine()

    engine.swap(build_process("coder"))
    second = engine.swap(build_process("judge"))

    assert second.active is True
    assert shutdown.shutdowns == ["coder"]
    assert engine.active_role == "judge"


def test_runtime_swap_sequence():
    engine, launcher, shutdown = make_engine()

    engine.swap(build_process("coder"))
    engine.swap(build_process("judge"))
    engine.swap(build_process("synth"))

    assert shutdown.shutdowns == ["coder", "judge"]
    assert engine.active_role == "synth"
    assert engine.swap_count == 2


def test_runtime_swap_history_roles():
    engine, launcher, shutdown = make_engine()

    engine.swap(build_process("coder"))
    engine.swap(build_process("judge"))
    engine.swap(build_process("synth"))

    assert engine.history_roles() == ["coder", "judge"]


def test_shutdown_active_no_replacement():
    engine, launcher, shutdown = make_engine()

    engine.swap(build_process("coder"))
    engine.shutdown_active()

    assert engine.active_process is None
    assert engine.has_active is False
    assert shutdown.shutdowns == ["coder"]


def test_shutdown_active_noop_when_none():
    engine, _, shutdown = make_engine()
    engine.shutdown_active()
    assert shutdown.shutdowns == []


def test_has_active_false_initially():
    engine, _, _ = make_engine()
    assert engine.has_active is False


def test_has_active_true_after_swap():
    engine, _, _ = make_engine()
    engine.swap(build_process("coder"))
    assert engine.has_active is True


def test_swap_count_increments():
    engine, _, _ = make_engine()
    assert engine.swap_count == 0
    engine.swap(build_process("coder"))
    engine.swap(build_process("judge"))
    assert engine.swap_count == 1


def test_active_role_none_initially():
    engine, _, _ = make_engine()
    assert engine.active_role is None