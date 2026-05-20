import pytest
from backend.runtime.runtime_bootstrap import RuntimeBootstrap, RuntimeBootstrapError
from backend.runtime.runtime_process import RuntimeProcess
from backend.runtime.runtime_session import RuntimeSession
from backend.runtime.runtime_teardown import RuntimeTeardown, RuntimeTeardownError


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeLauncher:
    def __init__(self) -> None:
        self.launched: list[str] = []

    def launch(self, process: RuntimeProcess) -> RuntimeProcess:
        process.mark_launched(pid=123)
        self.launched.append(process.role)
        return process


class FakeShutdown:
    def __init__(self) -> None:
        self.shutdowns: list[str] = []

    def shutdown(self, process: RuntimeProcess) -> None:
        process.mark_stopped()
        self.shutdowns.append(process.role)


class FakeHealth:
    def __init__(
        self,
        ready: bool = True,
        already_up: bool = False,
        stops: bool = True,
    ) -> None:
        self._ready = ready
        self._already_up = already_up
        self._stops = stops

    def wait_until_ready(
        self,
        *,
        port: int,
        model_name: str | None = None,
        timeout: int = 300,
    ) -> bool:
        return self._ready

    def is_ready(self, port: int, model_name: str | None = None) -> bool:
        return self._already_up

    def wait_until_stopped(self, *, port: int, timeout: int = 30) -> bool:
        return self._stops


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_process(role: str = "coder", port: int = 8000) -> RuntimeProcess:
    return RuntimeProcess(
        role=role,
        model_path=f"fake-{role}-path",
        model_name=f"fake-{role}",
        port=port,
    )


def make_bootstrap(
    ready: bool = True,
    already_up: bool = False,
) -> tuple[RuntimeBootstrap, FakeLauncher]:
    launcher = FakeLauncher()
    health = FakeHealth(ready=ready, already_up=already_up)
    bootstrap = RuntimeBootstrap(launcher=launcher, health=health)
    return bootstrap, launcher


# ── RuntimeBootstrap ──────────────────────────────────────────────────────────

def test_boot_success():
    bootstrap, launcher = make_bootstrap()
    process = build_process()

    launched = bootstrap.boot(process)

    assert launched.active is True
    assert launched.pid == 123
    assert launcher.launched == ["coder"]


def test_boot_returns_process():
    bootstrap, _ = make_bootstrap()
    process = build_process()
    launched = bootstrap.boot(process)
    assert launched is process


def test_boot_already_running_raises():
    bootstrap, _ = make_bootstrap()
    process = build_process()
    process.mark_launched(pid=999)

    with pytest.raises(RuntimeBootstrapError, match="already running"):
        bootstrap.boot(process)


def test_boot_port_already_in_use_raises():
    bootstrap, _ = make_bootstrap(already_up=True)
    process = build_process()

    with pytest.raises(RuntimeBootstrapError, match="already in use"):
        bootstrap.boot(process)


def test_boot_timeout_raises():
    bootstrap, _ = make_bootstrap(ready=False)
    process = build_process()

    with pytest.raises(RuntimeBootstrapError, match="failed to become ready"):
        bootstrap.boot(process)


def test_boot_and_verify_success():
    launcher = FakeLauncher()
    health = FakeHealth(ready=True, already_up=False)

    # After launch, is_ready should return True for verify ping
    call_count = [0]
    original_is_ready = health.is_ready

    def patched_is_ready(port: int, model_name: str | None = None) -> bool:
        call_count[0] += 1
        # First call = pre-boot check (returns False = port free)
        # Second call = post-boot verify (returns True = ready)
        return call_count[0] > 1

    health.is_ready = patched_is_ready
    bootstrap = RuntimeBootstrap(launcher=launcher, health=health)
    process = build_process()

    launched = bootstrap.boot_and_verify(process)
    assert launched.active is True


# ── RuntimeTeardown ───────────────────────────────────────────────────────────

def test_teardown_active_process():
    shutdown = FakeShutdown()
    teardown = RuntimeTeardown(
        shutdown=shutdown,
        health=FakeHealth(stops=True),
    )
    process = build_process()
    process.mark_launched(pid=999)

    teardown.teardown(process)

    assert process.active is False
    assert "coder" in shutdown.shutdowns


def test_teardown_inactive_process_noop():
    shutdown = FakeShutdown()
    teardown = RuntimeTeardown(shutdown=shutdown)
    process = build_process()
    # Never launched — active is False

    teardown.teardown(process)

    assert shutdown.shutdowns == []


def test_teardown_timeout_raises():
    shutdown = FakeShutdown()
    teardown = RuntimeTeardown(
        shutdown=shutdown,
        health=FakeHealth(stops=False),
        stop_timeout=30,
    )
    process = build_process()
    process.mark_launched(pid=999)

    with pytest.raises(RuntimeTeardownError, match="did not stop"):
        teardown.teardown(process)


def test_teardown_all_returns_results():
    shutdown = FakeShutdown()
    teardown = RuntimeTeardown(
        shutdown=shutdown,
        health=FakeHealth(stops=True),
    )
    p1 = build_process("coder", port=8000)
    p1.mark_launched(pid=1)
    p2 = build_process("judge", port=8001)
    p2.mark_launched(pid=2)

    results = teardown.teardown_all([p1, p2])

    assert results["coder"] is True
    assert results["judge"] is True


def test_teardown_all_partial_failure():
    shutdown = FakeShutdown()

    call_count = [0]

    class FlakeyHealth:
        def is_ready(self, port): return False
        def wait_until_stopped(self, *, port, timeout=30):
            call_count[0] += 1
            return call_count[0] > 1  # First call fails, second succeeds

    teardown = RuntimeTeardown(
        shutdown=shutdown,
        health=FlakeyHealth(),
    )
    p1 = build_process("coder", port=8000)
    p1.mark_launched(pid=1)
    p2 = build_process("judge", port=8001)
    p2.mark_launched(pid=2)

    results = teardown.teardown_all([p1, p2])

    assert results["coder"] is False
    assert results["judge"] is True


# ── RuntimeSession ────────────────────────────────────────────────────────────

def test_session_creation():
    process = build_process()
    session = RuntimeSession(process=process, objective="Improve auth")

    assert session.role == "coder"
    assert session.is_active is True
    assert session.duration_seconds is None


def test_session_end():
    process = build_process()
    session = RuntimeSession(process=process, objective="Improve auth")

    session.end()

    assert session.is_active is False
    assert session.duration_seconds is not None
    assert session.duration_seconds >= 0


def test_session_end_twice_raises():
    process = build_process()
    session = RuntimeSession(process=process, objective="Improve auth")
    session.end()

    with pytest.raises(ValueError, match="already ended"):
        session.end()


def test_session_blank_objective_raises():
    process = build_process()

    with pytest.raises(ValueError, match="blank"):
        RuntimeSession(process=process, objective="   ")


def test_session_to_dict():
    process = build_process()
    process.mark_launched(pid=42)
    session = RuntimeSession(
        process=process,
        objective="Improve auth",
        metadata={"model": "qwen"},
    )
    session.end()

    d = session.to_dict()

    assert d["role"] == "coder"
    assert d["objective"] == "Improve auth"
    assert d["pid"] == 42
    assert d["port"] == 8000
    assert d["duration_seconds"] is not None
    assert d["metadata"]["model"] == "qwen"
