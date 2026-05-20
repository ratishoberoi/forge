import pytest
import json
from backend.runtime.artifact_exchange import ArtifactExchange
from backend.runtime.artifact_loader import ArtifactLoader
from backend.runtime.artifact_store import ArtifactStore
from backend.runtime.autonomous_courtroom import (
    AutonomousCourtroom,
    AutonomousCourtroomError,
)
from backend.runtime.runtime_process import RuntimeProcess
from backend.runtime.runtime_swap_engine import RuntimeSwapEngine


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeLauncher:
    def __init__(self) -> None:
        self.launched: list[str] = []

    def launch(self, process: RuntimeProcess) -> RuntimeProcess:
        process.mark_launched(pid=999)
        self.launched.append(process.role)
        return process


class FakeShutdown:
    def __init__(self) -> None:
        self.shutdowns: list[str] = []

    def shutdown(self, process: RuntimeProcess) -> None:
        process.mark_stopped()
        self.shutdowns.append(process.role)


class FakeInference:
    """
    Returns deterministic content based on model name.
    Model name is extracted as last path segment — matches
    _model_name_for_role() in AutonomousCourtroom.
    """
    def infer(
        self,
        *,
        port: int,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        system_prompt: str | None = None,
        response_format: dict | None = None,
    ) -> str:
        if model == "qwen-primary":
            return json.dumps({
                "summary": "fake primary response",
                "files": {"app.py": "print('hello')\n"},
            })
        if model == "deepseek-synth":
            return json.dumps({
                "critique": "fake synth response",
                "risks": ["low coverage"],
                "recommended_changes": ["add tests"],
            })
        return json.dumps({
            "verdict": "fake judge response",
            "approved": True,
            "required_changes": [],
        })


class FakeHealth:
    """Always reports runtime ready immediately — no real port polling."""
    def wait_until_ready(
        self,
        *,
        port: int,
        model_name: str | None = None,
        timeout: int = 300,
    ) -> bool:
        return True

    def is_ready(self, port: int) -> bool:
        return True


class FlakyInference(FakeInference):
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return "thinking before malformed output"
        return super().infer(**kwargs)


class SchemaRepairInference(FakeInference):
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return "Thinking Process:\n1. Analyze request\n\n{\"summary\": \"\", \"files\": {}}"
        return super().infer(**kwargs)


class RecordingHealth(FakeHealth):
    def __init__(self) -> None:
        self.ready_ports: list[int] = []

    def wait_until_ready(
        self,
        *,
        port: int,
        model_name: str | None = None,
        timeout: int = 300,
    ) -> bool:
        self.ready_ports.append(port)
        return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_swap_engine() -> tuple[RuntimeSwapEngine, FakeLauncher, FakeShutdown]:
    launcher = FakeLauncher()
    shutdown = FakeShutdown()
    engine = RuntimeSwapEngine(
        launcher=launcher,
        shutdown=shutdown,
        swap_delay_seconds=0,  # instant in tests
    )
    return engine, launcher, shutdown


def make_exchange(tmp_path) -> ArtifactExchange:
    return ArtifactExchange(
        store=ArtifactStore(str(tmp_path)),
        loader=ArtifactLoader(str(tmp_path)),
    )


def make_courtroom(
    tmp_path,
) -> tuple[AutonomousCourtroom, FakeLauncher, FakeShutdown]:
    engine, launcher, shutdown = make_swap_engine()
    courtroom = AutonomousCourtroom(
        swap_engine=engine,
        exchange=make_exchange(tmp_path),
        inference=FakeInference(),
    )
    courtroom.health = FakeHealth()
    return courtroom, launcher, shutdown


# ── execute ───────────────────────────────────────────────────────────────────

def test_autonomous_courtroom_returns_three_artifacts(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    artifacts = courtroom.execute(objective="Improve auth")
    assert len(artifacts) == 3


def test_autonomous_courtroom_role_order(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    artifacts = courtroom.execute(objective="Improve auth")
    assert artifacts[0].role == "PRIMARY_CODER"
    assert artifacts[1].role == "DEEPSEEK_SYNTH"
    assert artifacts[2].role == "JUDGE"


def test_autonomous_courtroom_swap_sequence(tmp_path):
    courtroom, launcher, shutdown = make_courtroom(tmp_path)
    courtroom.execute(objective="Improve auth")

    assert launcher.launched == [
        "PRIMARY_CODER",
        "DEEPSEEK_SYNTH",
        "JUDGE",
    ]
    assert "PRIMARY_CODER" in shutdown.shutdowns
    assert "DEEPSEEK_SYNTH" in shutdown.shutdowns
    assert "JUDGE" in shutdown.shutdowns


def test_autonomous_courtroom_all_on_same_port(tmp_path):
    """All three runtimes must launch on the same shared port."""
    courtroom, launcher, _ = make_courtroom(tmp_path)
    courtroom.execute(objective="Improve auth")

    # All processes created with same port
    assert courtroom.port == 8000


def test_autonomous_courtroom_artifacts_persisted(tmp_path):
    exchange = make_exchange(tmp_path)
    engine, _, _ = make_swap_engine()
    courtroom = AutonomousCourtroom(
        swap_engine=engine,
        exchange=exchange,
        inference=FakeInference(),
    )
    courtroom.health = FakeHealth()

    courtroom.execute(objective="Improve auth")

    assert exchange.exists("PRIMARY_CODER", 1)
    assert exchange.exists("DEEPSEEK_SYNTH", 1)
    assert exchange.exists("JUDGE", 1)


def test_autonomous_courtroom_artifact_ids(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    artifacts = courtroom.execute(objective="Improve auth")

    assert artifacts[0].artifact_id == "coder_round_1"
    assert artifacts[1].artifact_id == "synth_round_1"
    assert artifacts[2].artifact_id == "judge_round_1"


def test_autonomous_courtroom_artifact_content(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    artifacts = courtroom.execute(objective="Improve auth")

    assert '"files"' in artifacts[0].content
    assert '"critique"' in artifacts[1].content
    assert '"approved": true' in artifacts[2].content
    assert all(a.metadata["schema_valid"] is True for a in artifacts)


def test_autonomous_courtroom_retries_malformed_output(tmp_path):
    engine, _, _ = make_swap_engine()
    inference = FlakyInference()
    courtroom = AutonomousCourtroom(
        swap_engine=engine,
        exchange=make_exchange(tmp_path),
        inference=inference,
    )
    courtroom.health = FakeHealth()

    artifacts = courtroom.execute(objective="Improve auth")

    assert len(artifacts) == 3
    assert inference.calls == 4


def test_autonomous_courtroom_repairs_malformed_schema_without_relaunch(tmp_path):
    engine, launcher, _ = make_swap_engine()
    inference = SchemaRepairInference()
    telemetry: list[str] = []
    courtroom = AutonomousCourtroom(
        swap_engine=engine,
        exchange=make_exchange(tmp_path),
        inference=inference,
        telemetry=telemetry.append,
    )
    courtroom.health = FakeHealth()

    artifacts = courtroom.execute(objective="Improve auth")

    assert len(artifacts) == 3
    assert inference.calls == 4
    assert launcher.launched == ["PRIMARY_CODER", "DEEPSEEK_SYNTH", "JUDGE"]
    assert any("[SCHEMA_REPAIR] PRIMARY_CODER" in line for line in telemetry)
    assert any("[SCHEMA_RETRY] PRIMARY_CODER" in line for line in telemetry)


def test_autonomous_courtroom_synth_references_coder(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    artifacts = courtroom.execute(objective="Improve auth")
    assert artifacts[1].metadata["critiques_artifact"] == "coder_round_1"


def test_autonomous_courtroom_judge_references_both(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    artifacts = courtroom.execute(objective="Improve auth")
    assert artifacts[2].metadata["reviews_coder"] == "coder_round_1"
    assert artifacts[2].metadata["reviews_synth"] == "synth_round_1"


def test_autonomous_courtroom_round_ids(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    artifacts = courtroom.execute(objective="Improve auth", round_id=2)

    assert all(a.round_id == 2 for a in artifacts)
    assert artifacts[0].artifact_id == "coder_round_2"


def test_autonomous_courtroom_blank_objective_raises(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    with pytest.raises(AutonomousCourtroomError, match="blank"):
        courtroom.execute(objective="   ")


def test_autonomous_courtroom_health_failure_raises(tmp_path):
    """If runtime does not become ready, AutonomousCourtroomError raised."""
    class NeverReadyHealth:
        def wait_until_ready(
            self,
            *,
            port: int,
            model_name: str | None = None,
            timeout: int = 300,
        ) -> bool:
            return False
        def is_ready(self, port: int) -> bool:
            return False

    engine, _, _ = make_swap_engine()
    courtroom = AutonomousCourtroom(
        swap_engine=engine,
        exchange=make_exchange(tmp_path),
        inference=FakeInference(),
    )
    courtroom.health = NeverReadyHealth()

    with pytest.raises(AutonomousCourtroomError, match="failed to become ready"):
        courtroom.execute(objective="Improve auth")


def test_autonomous_courtroom_recovers_after_runtime_readiness_restart(tmp_path):
    class FlakyReadyHealth:
        def __init__(self) -> None:
            self.calls = 0

        def wait_until_ready(
            self,
            *,
            port: int,
            model_name: str | None = None,
            timeout: int = 300,
        ) -> bool:
            self.calls += 1
            return self.calls > 1

        def is_ready(self, port: int) -> bool:
            return True

    engine, launcher, shutdown = make_swap_engine()
    courtroom = AutonomousCourtroom(
        swap_engine=engine,
        exchange=make_exchange(tmp_path),
        inference=FakeInference(),
        stage_attempts=2,
    )
    health = FlakyReadyHealth()
    courtroom.health = health

    artifacts = courtroom.execute(objective="Improve auth")

    assert len(artifacts) == 3
    assert health.calls == 4
    assert launcher.launched[:2] == ["PRIMARY_CODER", "PRIMARY_CODER"]
    assert "PRIMARY_CODER" in shutdown.shutdowns


def test_autonomous_courtroom_uses_reassigned_runtime_port(tmp_path):
    class ReassigningLauncher(FakeLauncher):
        def launch(self, process: RuntimeProcess) -> RuntimeProcess:
            process.port = 49152
            return super().launch(process)

    launcher = ReassigningLauncher()
    shutdown = FakeShutdown()
    engine = RuntimeSwapEngine(
        launcher=launcher,
        shutdown=shutdown,
        swap_delay_seconds=0,
    )
    inference = FakeInference()
    courtroom = AutonomousCourtroom(
        swap_engine=engine,
        exchange=make_exchange(tmp_path),
        inference=inference,
        port=8000,
    )
    health = RecordingHealth()
    courtroom.health = health

    artifacts = courtroom.execute(objective="Improve auth")

    assert len(artifacts) == 3
    assert health.ready_ports == [49152, 49152, 49152]
    assert courtroom.port == 49152


def test_autonomous_courtroom_syncs_context_window_to_runtime_fallback(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    process = RuntimeProcess(
        role="PRIMARY_CODER",
        model_path="fake",
        model_name="qwen-primary",
        port=8000,
        metadata={"runtime_diagnostics": {"profile": {"max_model_len": 2048}}},
    )

    courtroom._sync_context_window_from_runtime(process)

    assert courtroom.context_window == 2048


# ── execute_multi_round ───────────────────────────────────────────────────────

def test_execute_multi_round_returns_correct_structure(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    all_rounds = courtroom.execute_multi_round(
        objective="Improve auth", rounds=3
    )
    assert len(all_rounds) == 3
    for round_artifacts in all_rounds:
        assert len(round_artifacts) == 3


def test_execute_multi_round_sequential_round_ids(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    all_rounds = courtroom.execute_multi_round(
        objective="Improve auth", rounds=2
    )
    assert all_rounds[0][0].round_id == 1
    assert all_rounds[1][0].round_id == 2


def test_execute_multi_round_zero_raises(tmp_path):
    courtroom, _, _ = make_courtroom(tmp_path)
    with pytest.raises(AutonomousCourtroomError, match="rounds"):
        courtroom.execute_multi_round(objective="Improve auth", rounds=0)


# ── Retrieve after execute ────────────────────────────────────────────────────

def test_retrieve_coder_artifact_after_execute(tmp_path):
    exchange = make_exchange(tmp_path)
    engine, _, _ = make_swap_engine()
    courtroom = AutonomousCourtroom(
        swap_engine=engine,
        exchange=exchange,
        inference=FakeInference(),
    )
    courtroom.health = FakeHealth()
    courtroom.execute(objective="Improve auth")

    coder = exchange.retrieve_round(role="PRIMARY_CODER", round_id=1)
    assert '"summary": "fake primary response"' in coder.content


def test_retrieve_full_role_history_after_multi_round(tmp_path):
    exchange = make_exchange(tmp_path)
    engine, _, _ = make_swap_engine()
    courtroom = AutonomousCourtroom(
        swap_engine=engine,
        exchange=exchange,
        inference=FakeInference(),
    )
    courtroom.health = FakeHealth()
    courtroom.execute_multi_round(objective="Improve auth", rounds=3)

    history = exchange.retrieve_role_history("PRIMARY_CODER")
    assert len(history) == 3
    assert [a.round_id for a in history] == [1, 2, 3]
