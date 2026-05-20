from __future__ import annotations

import os
import json
import subprocess
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.api.routes import control_center
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_store import ArtifactStore
from backend.runtime.run_history import RunHistoryStore
from backend.runtime.workspace_manager import WorkspaceManager
from backend.config.settings import Settings


def test_control_center_snapshot_shape(tmp_path) -> None:
    app = create_app(start_runtime=False)
    client = TestClient(app)

    response = client.get(
        "/api/control/snapshot",
        params={"repository_root": str(tmp_path), "artifact_dir": str(tmp_path / "artifacts")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["runtime"]["health"] in {"idle", "running"}
    assert isinstance(body["courtroom"], list)
    assert [role["role"] for role in body["courtroom"]] == [
        "PRIMARY_CODER",
        "DEEPSEEK_SYNTH",
        "JUDGE",
    ]
    assert isinstance(body["timeline"], list)
    assert isinstance(body["logs"], list)
    assert body["repository_summary"] is not None
    assert body["execution_plan"] is not None
    assert isinstance(body["convergence"], dict)
    assert isinstance(body["repositories"], list)
    assert isinstance(body["run_history"], list)
    assert isinstance(body["queued_tasks"], list)


def test_control_center_repository_tree_and_file(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")

    app = create_app(start_runtime=False)
    client = TestClient(app)

    tree_response = client.get("/api/control/repository/tree", params={"root": str(repo)})
    assert tree_response.status_code == 200
    tree = tree_response.json()
    assert any(child["path"] == "app.py" for child in tree["children"])

    file_response = client.get(
        "/api/control/repository/file",
        params={"root": str(repo), "path": "app.py"},
    )
    assert file_response.status_code == 200
    assert file_response.json()["content"] == "print('hello')\n"


def test_control_center_run_lifecycle_without_execution(tmp_path) -> None:
    app = create_app(start_runtime=False)
    client = TestClient(app)

    response = client.post(
        "/api/control/runs",
        json={
            "objective": "Build a calculator app",
            "repository_root": str(tmp_path),
            "target_file": "calculator.py",
            "test_command": ["pytest", "-q"],
            "max_iterations": 1,
            "execute": False,
        },
    )
    assert response.status_code == 200
    run = response.json()
    assert run["status"] == "queued"

    pause = client.post(f"/api/control/runs/{run['id']}/pause")
    assert pause.status_code == 200
    assert pause.json()["status"] == "paused"

    resume = client.post(f"/api/control/runs/{run['id']}/resume")
    assert resume.status_code == 200
    assert resume.json()["status"] == "queued"

    stop = client.post(f"/api/control/runs/{run['id']}/stop")
    assert stop.status_code == 200
    assert stop.json()["status"] == "stopping"


def test_control_center_workspace_git_and_history_apis(tmp_path) -> None:
    repo = tmp_path / "repo"
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@test.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@test.com",
        "HOME": str(tmp_path),
        "PATH": os.environ["PATH"],
    }
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, env=env)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, env=env)
    state = control_center.ControlCenterState()
    state.workspace_manager = WorkspaceManager(
        registry_path=str(tmp_path / "registry.json"),
        workspace_root=str(tmp_path / "workspaces"),
        env=env,
    )
    state.run_history = RunHistoryStore(str(tmp_path / "runs.json"))
    control_center.router._control_state = state  # type: ignore[attr-defined]
    app = create_app(start_runtime=False)
    client = TestClient(app)

    imported = client.post(
        "/api/control/workspaces/import",
        json={"path": str(repo), "repository_name": "repo"},
    )
    repository = imported.json()
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    commit = client.post(
        "/api/control/git/commit",
        json={"repository_id": repository["repository_id"], "message": "update value"},
    )
    history = client.get(
        "/api/control/git/history",
        params={"repository_id": repository["repository_id"]},
    )
    rollback = client.post(
        "/api/control/git/rollback",
        json={"repository_id": repository["repository_id"], "target": "HEAD"},
    )

    assert imported.status_code == 200
    assert commit.status_code == 200
    assert commit.json()["commit_sha"]
    assert history.status_code == 200
    assert history.json()["commits"][0]["subject"] == "update value"
    assert rollback.status_code == 200


def test_control_center_workspace_browse_validate_and_import_refresh(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (repo / "app.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")
    state = control_center.ControlCenterState()
    state.workspace_manager = WorkspaceManager(
        registry_path=str(tmp_path / "registry.json"),
        workspace_root=str(tmp_path / "workspaces"),
        settings=Settings(
            repo_index_state_path=str(tmp_path / ".forge" / "index_state.json"),
            repo_incremental=False,
        ),
    )
    control_center.router._control_state = state  # type: ignore[attr-defined]
    app = create_app(start_runtime=False)
    client = TestClient(app)

    browse = client.get("/api/control/workspaces/browse", params={"path": str(tmp_path)})
    validate = client.post("/api/control/workspaces/validate", json={"path": str(repo)})
    invalid = client.post("/api/control/workspaces/validate", json={"path": str(tmp_path / "missing")})
    imported = client.post(
        "/api/control/workspaces/import",
        json={"path": str(repo), "repository_name": "demo", "refresh_intelligence": True},
    )

    assert browse.status_code == 200
    assert any(entry["path"] == str(repo) for entry in browse.json()["entries"])
    assert validate.status_code == 200
    assert validate.json()["valid"] is True
    assert invalid.status_code == 200
    assert invalid.json()["valid"] is False
    assert imported.status_code == 200
    body = imported.json()
    assert body["repository_path"] == str(repo)
    assert body["intelligence"] is not None
    assert body["metadata"]["language"] == "python"


def test_control_center_snapshot_uses_cached_preparation_for_stable_polling(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    state = control_center.ControlCenterState()
    control_center.router._control_state = state  # type: ignore[attr-defined]
    app = create_app(start_runtime=False)
    client = TestClient(app)

    first = client.get("/api/control/snapshot", params={"repository_root": str(tmp_path)})
    second = client.get("/api/control/snapshot", params={"repository_root": str(tmp_path)})
    health = client.get("/api/control/health")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["diagnostics"]["snapshot_cache_hits"] >= 1
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_snapshot_binds_to_active_workspace_repository(tmp_path) -> None:
    repo = tmp_path / "active"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='active'\n", encoding="utf-8")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    state = control_center.ControlCenterState()
    state.workspace_manager = WorkspaceManager(
        registry_path=str(tmp_path / "registry.json"),
        workspace_root=str(tmp_path / "workspaces"),
    )
    record = state.workspace_manager.import_local_repository(str(repo), repository_name="active")
    control_center.router._control_state = state  # type: ignore[attr-defined]
    app = create_app(start_runtime=False)
    client = TestClient(app)

    response = client.get("/api/control/snapshot")

    assert response.status_code == 200
    body = response.json()
    assert body["active_repository_id"] == record.repository_id
    assert body["active_repository_root"] == str(repo.resolve())
    assert body["repository_summary"]["root"] == str(repo.resolve())


def test_active_run_artifacts_are_isolated_from_stale_demo_artifacts(tmp_path) -> None:
    stale_dir = tmp_path / "runtime_artifacts"
    active_dir = tmp_path / "run-artifacts"
    _write_artifact(stale_dir, "PRIMARY_CODER", 1, "Improve authentication error handling in app.py")
    _write_artifact(active_dir, "PRIMARY_CODER", 1, "Build FastAPI Todo app")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    state = control_center.ControlCenterState()
    run = control_center.RunRecord(
        request=control_center.RunRequest(
            objective="Build FastAPI Todo app",
            repository_root=str(repo),
            target_file="app.py",
            artifact_dir=str(active_dir),
            execute=False,
        )
    )
    state._runs[run.id] = run
    control_center.router._control_state = state  # type: ignore[attr-defined]
    app = create_app(start_runtime=False)
    client = TestClient(app)

    response = client.get(
        "/api/control/snapshot",
        params={"repository_root": str(repo), "artifact_dir": str(stale_dir)},
    )

    assert response.status_code == 200
    artifacts = response.json()["artifacts"]
    assert artifacts
    assert artifacts[0]["task"] == "Build FastAPI Todo app"
    assert "Improve authentication" not in json.dumps(artifacts)


def test_control_center_run_state_machine_progresses_with_fake_runner(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    state = control_center.ControlCenterState()
    control_center.router._control_state = state  # type: ignore[attr-defined]

    class FakeAutonomousRun:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def execute_full(self, *, progress_callback, **kwargs):
            for phase in ("REPOSITORY_SCAN", "PLANNING", "CODER", "SYNTH", "JUDGE", "PATCH", "TESTS"):
                progress_callback(phase)
            return {
                "tests_passed": True,
                "repair_convergence": {"telemetry": ["[CONVERGED] tests_passed"]},
                "execution_branch": None,
            }

    monkeypatch.setattr(control_center, "AutonomousRun", FakeAutonomousRun)
    request = control_center.RunRequest(
        objective="Build FastAPI Todo app",
        repository_root=str(repo),
        target_file="app.py",
        execute=False,
    )
    record = state.create_run(request)
    state.execute_run(record.id)

    completed = state.get_run(record.id)
    assert completed.status == "completed"
    assert completed.phase == "CONVERGED"
    assert any("CODER" in item for item in completed.telemetry)
    assert any("CONVERGED" in item for item in completed.telemetry)


def test_execution_graph_updates_from_runtime_transitions(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    state = control_center.ControlCenterState()
    request = control_center.RunRequest(
        objective="Build FastAPI Todo app",
        repository_root=str(repo),
        target_file="app.py",
        execute=False,
    )
    record = state.create_run(request)
    record.status = "running"
    record.transition("REPOSITORY_SCAN")
    record.transition("PLANNING")
    record.transition("CODER")
    state._runs[record.id] = record
    control_center.router._control_state = state  # type: ignore[attr-defined]
    app = create_app(start_runtime=False)
    client = TestClient(app)

    running_snapshot = client.get("/api/control/snapshot", params={"repository_root": str(repo)}).json()
    record.transition("FAILED")
    failed_snapshot = client.get("/api/control/snapshot", params={"repository_root": str(repo)}).json()

    assert running_snapshot["running_node"]["step_id"] == "CODER"
    assert {node["step_id"] for node in running_snapshot["completed_nodes"]} == {"REPOSITORY_SCAN", "PLANNING"}
    assert running_snapshot["execution_graph"]["running"] == "CODER"
    assert any("[STEP_START] CODER" in item for item in record.telemetry)
    assert failed_snapshot["failed_nodes"][0]["step_id"] == "CODER"
    assert failed_snapshot["execution_graph"]["failed"] == ["CODER"]
    assert any("[STEP_FAILED] CODER" in item for item in record.telemetry)


def test_execution_graph_marks_repair_running_and_completed(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    request = control_center.RunRequest(
        objective="Repair failing tests",
        repository_root=str(repo),
        target_file="app.py",
        execute=False,
    )
    record = control_center.RunRecord(request=request)

    for phase in ("REPOSITORY_SCAN", "PLANNING", "CODER", "SYNTH", "JUDGE", "PATCH", "TESTS", "REPAIR"):
        record.transition(phase)
    graph = record.event_graph()

    assert graph["running"] == "REPAIR"
    assert "TESTS" in graph["completed"]
    record.transition("CONVERGED")
    graph = record.event_graph()
    assert "REPAIR" in graph["completed"]
    assert "CONVERGED" in graph["completed"]
    assert graph["running"] is None


def test_create_run_dispatches_execution_out_of_queued_state(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    state = control_center.ControlCenterState()
    control_center.router._control_state = state  # type: ignore[attr-defined]

    class FakeAutonomousRun:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def execute_full(self, *, progress_callback, **kwargs):
            for phase in ("REPOSITORY_SCAN", "PLANNING", "CODER", "SYNTH", "JUDGE", "PATCH", "TESTS"):
                progress_callback(phase)
            return {
                "tests_passed": True,
                "repair_convergence": {"telemetry": ["[TEST_PASS] attempt=0", "[CONVERGED] tests_passed"]},
                "execution_branch": None,
            }

    monkeypatch.setattr(control_center, "AutonomousRun", FakeAutonomousRun)
    app = create_app(start_runtime=False)
    client = TestClient(app)

    created = client.post(
        "/api/control/runs",
        json={
            "objective": "Build a complete FastAPI Todo application",
            "repository_root": str(repo),
            "target_file": "app/main.py",
            "execute": True,
        },
    )
    run_id = created.json()["id"]
    deadline = time.time() + 5
    latest = created.json()
    while time.time() < deadline:
        latest = client.get(f"/api/control/runs/{run_id}").json()
        if latest["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    snapshot = client.get("/api/control/snapshot", params={"repository_root": str(repo)}).json()

    assert created.status_code == 200
    assert latest["status"] == "completed"
    assert latest["phase"] == "CONVERGED"
    assert any("REPOSITORY_SCAN" in item for item in latest["telemetry"])
    assert snapshot["queued_tasks"] == []
    assert snapshot["active_run"] is None


def test_snapshot_uses_latest_run_objective_not_stale_inspect_repository(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = control_center.ControlCenterState()
    control_center.router._control_state = state  # type: ignore[attr-defined]
    app = create_app(start_runtime=False)
    client = TestClient(app)

    first = client.post(
        "/api/control/runs",
        json={
            "objective": "Inspect repository",
            "repository_root": str(repo),
            "target_file": "README.md",
            "execute": False,
        },
    )
    second = client.post(
        "/api/control/runs",
        json={
            "objective": "Build a complete FastAPI Todo application",
            "repository_root": str(repo),
            "target_file": "app/main.py",
            "execute": False,
        },
    )
    snapshot = client.get("/api/control/snapshot", params={"repository_root": str(repo)})

    assert first.status_code == 200
    assert second.status_code == 200
    assert snapshot.status_code == 200
    body = snapshot.json()
    plan = body["generated_plan"]
    assert body["active_objective"] == "Build a complete FastAPI Todo application"
    assert body["objective_source"] == "active_run"
    assert body["objective_classification"] == "APPLICATION"
    assert plan["objective"] == "Build a complete FastAPI Todo application"
    assert plan["objective_type"] == "APPLICATION"
    assert "Inspect repository" not in json.dumps(body["task_plan"])
    assert "app/main.py" in plan["files_to_create"]
    assert "app/models.py" in plan["files_to_create"]
    assert "app/database.py" in plan["files_to_create"]
    assert "app/schemas.py" in plan["files_to_create"]
    assert "app/repository.py" in plan["files_to_create"]
    assert "tests/test_todos.py" in plan["expected_tests"]


def test_execution_logs_are_prioritized_before_runtime_logs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runtime_logs").mkdir()
    (tmp_path / "runtime_logs" / "primary_coder_stderr.log").write_text(
        "CUDA graphs\nAPIServer startup\n",
        encoding="utf-8",
    )
    state = control_center.ControlCenterState()
    state.log("[RUN_START] run-1: Build app")
    state.log("[REPOSITORY_SCAN] run-1")

    lines = control_center._combined_logs(state)

    assert "[RUN_START]" in lines[0]
    assert "[REPOSITORY_SCAN]" in lines[1]
    assert any("CUDA graphs" in line for line in lines[2:])


def test_default_run_history_filters_pytest_records(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    store = RunHistoryStore()
    store.record_started(
        run_id="test-run",
        objective="Build test app",
        repository_path="/tmp/pytest-of-ratish/pytest-1/repo",
    )
    store.record_started(
        run_id="user-run",
        objective="Build real app",
        repository_path="/home/ratish/workspaces/repo",
    )

    runs = store.list_runs()

    assert [record.run_id for record in runs] == ["user-run"]


def _write_artifact(base_dir: Path, role: str, round_id: int, task: str) -> None:
    ArtifactStore(str(base_dir)).save(
        CognitionArtifact(
            artifact_id=f"{role}-{round_id}",
            role=role,
            round_id=round_id,
            task=task,
            content='{"summary":"ok","files":{"app.py":"VALUE = 1\\n"}}',
            metadata={},
        )
    )
