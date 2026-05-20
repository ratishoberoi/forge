from __future__ import annotations
import socket
import pytest
from backend.runtime.runtime_launcher import (
    RuntimeLaunchError,
    RuntimeLauncher,
    RuntimeMemorySnapshot,
)
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


def test_runtime_launcher_reassigns_occupied_port(monkeypatch):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    occupied_port = listener.getsockname()[1]
    launched_commands: list[list[str]] = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            launched_commands.append(command)
            self.pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr("subprocess.Popen", FakePopen)
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr(
        RuntimeLauncher,
        "_gpu_memory_snapshot",
        lambda self: RuntimeMemorySnapshot(
            total_mb=48000,
            used_mb=4000,
            free_mb=44000,
            source="test",
        ),
    )
    process = build_process("PRIMARY_CODER", port=occupied_port)

    try:
        launched = RuntimeLauncher(startup_wait=0, startup_failure_window=0).launch(process)
    finally:
        listener.close()

    assert launched.port != occupied_port
    assert launched.metadata["requested_port"] == occupied_port
    assert launched.metadata["port_reassigned"] is True
    assert "--port" in launched_commands[0]
    assert str(launched.port) in launched_commands[0]


def test_runtime_launcher_retries_when_startup_reports_address_in_use(monkeypatch, tmp_path):
    launched_commands: list[list[str]] = []

    class FakePopen:
        calls = 0

        def __init__(self, command, **kwargs):
            FakePopen.calls += 1
            launched_commands.append(command)
            self.pid = 12345 + FakePopen.calls
            self._return_code = 1 if FakePopen.calls == 1 else None
            if FakePopen.calls == 1:
                kwargs["stderr"].write(b"OSError: [Errno 98] Address already in use\n")
                kwargs["stderr"].flush()

        def poll(self):
            return self._return_code

    monkeypatch.setattr("subprocess.Popen", FakePopen)
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr(
        RuntimeLauncher,
        "_gpu_memory_snapshot",
        lambda self: RuntimeMemorySnapshot(
            total_mb=48000,
            used_mb=4000,
            free_mb=44000,
            source="test",
        ),
    )
    process = build_process("PRIMARY_CODER", port=8123)

    launched = RuntimeLauncher(
        startup_wait=0,
        startup_failure_window=0,
        log_dir=str(tmp_path),
    ).launch(process)

    assert launched.active is True
    assert FakePopen.calls == 2
    assert launched.port != 8123
    assert launched.metadata["requested_port"] == 8123
    assert launched.metadata["port_reassigned"] is True
    assert str(8123) in launched_commands[0]
    assert str(launched.port) in launched_commands[1]


def test_runtime_launcher_fallbacks_when_precheck_reports_insufficient_memory(
    monkeypatch,
    tmp_path,
):
    launched_commands: list[list[str]] = []
    diagnostics: list[dict[str, object]] = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            launched_commands.append(command)
            self.pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr("subprocess.Popen", FakePopen)
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr(
        RuntimeLauncher,
        "_gpu_memory_snapshot",
        lambda self: RuntimeMemorySnapshot(
            total_mb=11000,
            used_mb=500,
            free_mb=10500,
            source="test",
        ),
    )
    monkeypatch.setattr(RuntimeLauncher, "_estimate_model_size_mb", staticmethod(lambda path: 7000))
    process = build_process("PRIMARY_CODER", port=8124)

    launched = RuntimeLauncher(
        startup_wait=0,
        startup_failure_window=0,
        log_dir=str(tmp_path),
        diagnostics_callback=diagnostics.append,
    ).launch(process)

    command = launched_commands[0]
    assert launched.active is True
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert command[command.index("--max-num-seqs") + 1] == "16"
    assert "--enforce-eager" in command
    assert launched.metadata["runtime_diagnostics"]["fallback_status"] == "reduced_context_cache_recovery"
    assert any(item["load_status"] == "precheck" for item in diagnostics)


def test_runtime_launcher_fails_fast_when_all_profiles_exceed_memory(
    monkeypatch,
    tmp_path,
):
    popen_called = False

    class FakePopen:
        def __init__(self, command, **kwargs):
            nonlocal popen_called
            popen_called = True
            self.pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr("subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        RuntimeLauncher,
        "_gpu_memory_snapshot",
        lambda self: RuntimeMemorySnapshot(
            total_mb=24000,
            used_mb=23000,
            free_mb=1000,
            source="test",
        ),
    )
    monkeypatch.setattr(RuntimeLauncher, "_estimate_model_size_mb", staticmethod(lambda path: 7000))
    process = build_process("PRIMARY_CODER", port=8125)

    with pytest.raises(RuntimeLaunchError, match="gpu_memory_utilization|insufficient VRAM"):
        RuntimeLauncher(
            startup_wait=0,
            startup_failure_window=0,
            log_dir=str(tmp_path),
        ).launch(process)

    assert popen_called is False
    assert process.metadata["runtime_diagnostics"]["load_status"] == "precheck"
    assert process.metadata["runtime_diagnostics"]["failure_reason"]


def test_runtime_launcher_uses_configured_profile_when_memory_is_sufficient(
    monkeypatch,
    tmp_path,
):
    launched_commands: list[list[str]] = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            launched_commands.append(command)
            self.pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr("subprocess.Popen", FakePopen)
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr(
        RuntimeLauncher,
        "_gpu_memory_snapshot",
        lambda self: RuntimeMemorySnapshot(
            total_mb=48000,
            used_mb=4000,
            free_mb=44000,
            source="test",
        ),
    )
    monkeypatch.setattr(RuntimeLauncher, "_estimate_model_size_mb", staticmethod(lambda path: 7000))
    process = build_process("PRIMARY_CODER", port=8126)

    RuntimeLauncher(
        startup_wait=0,
        startup_failure_window=0,
        log_dir=str(tmp_path),
    ).launch(process)

    command = launched_commands[0]
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert command[command.index("--max-num-seqs") + 1] == "64"
    assert "--enforce-eager" not in command


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
