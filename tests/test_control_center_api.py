from __future__ import annotations

import os
import subprocess

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.api.routes import control_center
from backend.runtime.run_history import RunHistoryStore
from backend.runtime.workspace_manager import WorkspaceManager


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
