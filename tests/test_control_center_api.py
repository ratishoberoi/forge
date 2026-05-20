from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app


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
