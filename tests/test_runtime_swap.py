from __future__ import annotations
import pytest
from backend.runtime.runtime_process import RuntimeProcess
from backend.runtime.runtime_health import RuntimeHealth
from backend.runtime.runtime_swap_engine import RuntimeSwapEngine, RuntimeSwapError


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeLauncher:
    def __init__(self) -> None:
        self.launched: list[str] = []
        self.processes: list[RuntimeProcess] = []

    def launch(self, process: RuntimeProcess) -> RuntimeProcess:
        process.mark_launched(pid=999)
        self.launched.append(process.role)
        self.processes.append(process)
        return process


class FakeShutdown:
    def __init__(self) -> None:
        self.shutdowns: list[str] = []

    def shutdown(self, process: RuntimeProcess) -> None:
        process.mark_stopped()
        self.shutdowns.append(process.role)


class FailingLauncher:
    def launch(self, process: RuntimeProcess) -> RuntimeProcess:
        raise RuntimeError("launch failed")


class FailingShutdown:
    def shutdown(self, process: RuntimeProcess) -> None:
        raise RuntimeError("shutdown failed")


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_process(
    role: str,
    port: int = 8000,
    model_name: str | None = None,
    model_path: str | None = None,
) -> RuntimeProcess:
    return RuntimeProcess(
        role=role,
        model_path=model_path or f"fake-{role}-path",
        model_name=model_name or f"fake-{role}",
        port=port,
    )


def make_engine(
    launcher: FakeLauncher | None = None,
    shutdown: FakeShutdown | None = None,
) -> tuple[RuntimeSwapEngine, FakeLauncher, FakeShutdown]:
    launcher = launcher or FakeLauncher()
    shutdown = shutdown or FakeShutdown()
    engine = RuntimeSwapEngine(
        launcher=launcher,
        shutdown=shutdown,
        swap_delay_seconds=0,  # instant in tests
    )
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
        RuntimeProcess(role="coder", model_name="fake-model", model_path="fake-model-path", port=80)


def test_process_empty_role_raises():
    with pytest.raises(ValueError, match="role"):
        RuntimeProcess(role="", model_name="fake-model", model_path="fake-model-path", port=8000)


def test_process_to_dict():
    process = build_process("coder")
    d = process.to_dict()
    assert d["role"] == "coder"
    assert d["model_name"] == "fake-coder"
    assert d["model_path"] == "fake-coder-path"
    assert d["active"] is False
    assert "pgid" in d
    assert "launched_at" in d


def test_runtime_health_requires_valid_model_registry():
    valid = {"data": [{"id": "qwen-primary", "object": "model"}]}
    assert RuntimeHealth._valid_model_registry(valid, model_name="qwen-primary") is True
    assert RuntimeHealth._valid_model_registry({"data": []}) is False
    assert RuntimeHealth._valid_model_registry({"data": [{"object": "model"}]}) is False
    assert RuntimeHealth._valid_model_registry(valid, model_name="qwen-judge") is False


# ── RuntimeSwapEngine: swap ───────────────────────────────────────────────────

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
    engine.swap(build_process("PRIMARY_CODER", model_name="qwen-primary"))
    engine.swap(build_process("DEEPSEEK_SYNTH", model_name="deepseek-synth"))
    engine.swap(build_process("JUDGE", model_name="qwen-judge"))
    assert shutdown.shutdowns == ["PRIMARY_CODER", "DEEPSEEK_SYNTH"]
    assert launcher.launched == ["PRIMARY_CODER", "DEEPSEEK_SYNTH", "JUDGE"]
    assert [p.model_name for p in launcher.processes] == [
        "qwen-primary",
        "deepseek-synth",
        "qwen-judge",
    ]
    assert engine.active_role == "JUDGE"
    assert engine.swap_count == 2


def test_runtime_swap_history_roles():
    engine, launcher, shutdown = make_engine()
    engine.swap(build_process("coder"))
    engine.swap(build_process("judge"))
    engine.swap(build_process("synth"))
    assert engine.history_roles() == ["coder", "judge"]


def test_swap_first_process_no_shutdown():
    engine, _, shutdown = make_engine()
    engine.swap(build_process("coder"))
    assert shutdown.shutdowns == []


def test_swap_launch_failure_raises_swap_error():
    engine = RuntimeSwapEngine(
        launcher=FailingLauncher(),
        shutdown=FakeShutdown(),
        swap_delay_seconds=0,
    )
    with pytest.raises(RuntimeSwapError, match="Failed to launch"):
        engine.swap(build_process("coder"))


def test_swap_shutdown_failure_raises_swap_error():
    engine = RuntimeSwapEngine(
        launcher=FakeLauncher(),
        shutdown=FailingShutdown(),
        swap_delay_seconds=0,
    )
    engine.swap(build_process("coder"))
    with pytest.raises(RuntimeSwapError, match="Failed to shutdown"):
        engine.swap(build_process("judge"))


def test_swap_delay_zero_is_instant():
    """swap_delay_seconds=0 — tests run instantly."""
    import time
    engine, _, _ = make_engine()
    start = time.monotonic()
    engine.swap(build_process("coder"))
    engine.swap(build_process("judge"))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


def test_swap_delay_respected():
    """swap_delay_seconds is actually passed and stored."""
    engine = RuntimeSwapEngine(
        launcher=FakeLauncher(),
        shutdown=FakeShutdown(),
        swap_delay_seconds=5.0,
    )
    assert engine.swap_delay_seconds == 5.0


# ── RuntimeSwapEngine: shutdown_active ───────────────────────────────────────

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


def test_shutdown_active_clears_active_even_on_error():
    engine = RuntimeSwapEngine(
        launcher=FakeLauncher(),
        shutdown=FailingShutdown(),
        swap_delay_seconds=0,
    )
    engine.swap(build_process("coder"))
    try:
        engine.shutdown_active()
    except Exception:
        pass
    assert engine.active_process is None


# ── RuntimeSwapEngine: properties ────────────────────────────────────────────

def test_has_active_false_initially():
    engine, _, _ = make_engine()
    assert engine.has_active is False


def test_has_active_true_after_swap():
    engine, _, _ = make_engine()
    engine.swap(build_process("coder"))
    assert engine.has_active is True


def test_swap_count_zero_initially():
    engine, _, _ = make_engine()
    assert engine.swap_count == 0


def test_swap_count_increments_on_second_swap():
    engine, _, _ = make_engine()
    engine.swap(build_process("coder"))
    engine.swap(build_process("judge"))
    assert engine.swap_count == 1


def test_swap_count_after_shutdown_active():
    engine, _, _ = make_engine()
    engine.swap(build_process("coder"))
    engine.shutdown_active()
    assert engine.swap_count == 1


def test_active_role_none_initially():
    engine, _, _ = make_engine()
    assert engine.active_role is None


def test_active_role_after_swap():
    engine, _, _ = make_engine()
    engine.swap(build_process("coder"))
    assert engine.active_role == "coder"


def test_active_role_none_after_shutdown():
    engine, _, _ = make_engine()
    engine.swap(build_process("coder"))
    engine.shutdown_active()
    assert engine.active_role is None
