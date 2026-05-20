"""Operator dashboard routes for Forge Control Center."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config.settings import Settings, get_settings
from backend.runtime.artifact_loader import ArtifactLoader
from backend.runtime.autonomous_run import AutonomousRun
from backend.runtime.git_diff import GitDiff, GitDiffError


RunStatus = Literal[
    "queued",
    "running",
    "paused",
    "stopping",
    "completed",
    "failed",
    "cancelled",
]

COURTROOM_ROLES = ("PRIMARY_CODER", "DEEPSEEK_SYNTH", "JUDGE")
TIMELINE_STAGES = (
    "Repository Scan",
    "Planning",
    "Coder",
    "Synth",
    "Judge",
    "Patch",
    "Tests",
    "Repair",
    "Converged",
)


class RunRequest(BaseModel):
    objective: str = Field(min_length=1)
    repository_root: str
    target_file: str
    test_command: list[str] = Field(default_factory=lambda: ["pytest", "-q"])
    max_iterations: int = Field(default=3, ge=1, le=20)
    artifact_dir: str = "runtime_artifacts"
    inference_timeout: int = Field(default=180, ge=1)
    execute: bool = True
    models: dict[str, str] = Field(default_factory=dict)


class RunSummary(BaseModel):
    id: str
    objective: str
    repository_root: str
    target_file: str
    test_command: list[str]
    max_iterations: int
    status: RunStatus
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    stop_requested: bool = False
    pause_requested: bool = False
    result: dict | None = None


class ArtifactSummary(BaseModel):
    artifact_id: str
    role: str
    round_id: int
    task: str
    content: str
    created_at: str
    metadata: dict


class RepoTreeNode(BaseModel):
    name: str
    path: str
    type: Literal["file", "directory"]
    size_bytes: int | None = None
    children: list["RepoTreeNode"] = Field(default_factory=list)


class ControlCenterSnapshot(BaseModel):
    generated_at: str
    active_run: RunSummary | None
    runs: list[RunSummary]
    courtroom: list[dict[str, object]]
    timeline: list[dict[str, object]]
    runtime: dict[str, object]
    artifacts: list[ArtifactSummary]
    patch: dict[str, object]
    tests: dict[str, object]
    logs: list[str]
    conversation: list[dict[str, object]]


@dataclass(slots=True)
class RunRecord:
    request: RunRequest
    id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex}")
    status: RunStatus = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict | None = None
    stop_requested: bool = False
    pause_requested: bool = False

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def summary(self) -> RunSummary:
        return RunSummary(
            id=self.id,
            objective=self.request.objective,
            repository_root=self.request.repository_root,
            target_file=self.request.target_file,
            test_command=self.request.test_command,
            max_iterations=self.request.max_iterations,
            status=self.status,
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            started_at=self.started_at.isoformat() if self.started_at else None,
            completed_at=self.completed_at.isoformat() if self.completed_at else None,
            error=self.error,
            stop_requested=self.stop_requested,
            pause_requested=self.pause_requested,
            result=self.result,
        )


class ControlCenterState:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._runs: dict[str, RunRecord] = {}
        self._logs: deque[str] = deque(maxlen=500)
        self._lock = Lock()

    def create_run(self, request: RunRequest) -> RunRecord:
        record = RunRecord(request=request)
        with self._lock:
            self._runs[record.id] = record
            self.log(f"[RUN] queued {record.id}: {request.objective}")
        return record

    def get_run(self, run_id: str) -> RunRecord:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            return record

    def runs(self) -> list[RunRecord]:
        with self._lock:
            return sorted(
                self._runs.values(),
                key=lambda record: record.created_at,
                reverse=True,
            )

    def active_run(self) -> RunRecord | None:
        with self._lock:
            for record in sorted(
                self._runs.values(),
                key=lambda item: item.updated_at,
                reverse=True,
            ):
                if record.status in {"queued", "running", "paused", "stopping"}:
                    return record
        return None

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self._logs.append(f"{timestamp} {message}")

    def logs(self) -> list[str]:
        return list(self._logs)

    def pause(self, run_id: str) -> RunRecord:
        record = self.get_run(run_id)
        record.pause_requested = True
        if record.status in {"queued", "running"}:
            record.status = "paused"
        record.touch()
        self.log(f"[RUN] pause requested {run_id}")
        return record

    def resume(self, run_id: str) -> RunRecord:
        record = self.get_run(run_id)
        record.pause_requested = False
        if record.status == "paused":
            record.status = "queued"
        record.touch()
        self.log(f"[RUN] resume requested {run_id}")
        return record

    def stop(self, run_id: str) -> RunRecord:
        record = self.get_run(run_id)
        record.stop_requested = True
        if record.status in {"queued", "running", "paused"}:
            record.status = "stopping"
        record.touch()
        self.log(f"[RUN] stop requested {run_id}")
        return record

    def execute_run(self, run_id: str) -> None:
        record = self.get_run(run_id)
        if record.stop_requested:
            record.status = "cancelled"
            record.completed_at = datetime.now(timezone.utc)
            record.touch()
            self.log(f"[RUN] cancelled before start {run_id}")
            return

        record.status = "running"
        record.started_at = datetime.now(timezone.utc)
        record.touch()
        self.log(f"[RUN] started {run_id}")
        try:
            runner = AutonomousRun(
                repo_root=record.request.repository_root,
                artifact_dir=record.request.artifact_dir,
                inference_timeout=record.request.inference_timeout,
            )
            result = runner.execute_full(
                objective=record.request.objective,
                target_file=record.request.target_file,
                test_command=record.request.test_command,
                max_iterations=record.request.max_iterations,
            )
            record.result = result
            record.status = "completed"
            self.log(f"[RUN] completed {run_id}")
        except Exception as exc:
            record.error = str(exc)
            record.status = "failed"
            self.log(f"[RUN] failed {run_id}: {exc}")
        finally:
            record.completed_at = datetime.now(timezone.utc)
            record.touch()


router = APIRouter(prefix="/api/control", tags=["control-center"])


def get_control_state() -> ControlCenterState:
    if not hasattr(router, "_control_state"):
        router._control_state = ControlCenterState()  # type: ignore[attr-defined]
    return router._control_state  # type: ignore[attr-defined]


@router.get("/snapshot", response_model=ControlCenterSnapshot)
async def snapshot(
    repository_root: str | None = None,
    artifact_dir: str = "runtime_artifacts",
) -> ControlCenterSnapshot:
    state = get_control_state()
    active = state.active_run()
    repo_root = repository_root or (active.request.repository_root if active else os.getcwd())
    artifacts = _load_artifacts(artifact_dir)
    latest_result = _latest_result(state)
    return ControlCenterSnapshot(
        generated_at=datetime.now(timezone.utc).isoformat(),
        active_run=active.summary() if active else None,
        runs=[record.summary() for record in state.runs()],
        courtroom=_courtroom_state(active),
        timeline=_timeline_state(active),
        runtime=_runtime_snapshot(active),
        artifacts=artifacts,
        patch=_patch_snapshot(repo_root),
        tests=_tests_snapshot(latest_result),
        logs=_combined_logs(state),
        conversation=_conversation_summaries(artifacts),
    )


@router.post("/runs", response_model=RunSummary)
async def create_run(request: RunRequest, background_tasks: BackgroundTasks) -> RunSummary:
    repo_root = Path(request.repository_root).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise HTTPException(status_code=400, detail=f"repository_root does not exist: {repo_root}")
    state = get_control_state()
    record = state.create_run(request.model_copy(update={"repository_root": str(repo_root)}))
    if request.execute:
        background_tasks.add_task(state.execute_run, record.id)
    return record.summary()


@router.get("/runs", response_model=list[RunSummary])
async def list_runs() -> list[RunSummary]:
    return [record.summary() for record in get_control_state().runs()]


@router.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(run_id: str) -> RunSummary:
    try:
        return get_control_state().get_run(run_id).summary()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}") from exc


@router.post("/runs/{run_id}/pause", response_model=RunSummary)
async def pause_run(run_id: str) -> RunSummary:
    try:
        return get_control_state().pause(run_id).summary()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}") from exc


@router.post("/runs/{run_id}/resume", response_model=RunSummary)
async def resume_run(run_id: str) -> RunSummary:
    try:
        return get_control_state().resume(run_id).summary()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}") from exc


@router.post("/runs/{run_id}/stop", response_model=RunSummary)
async def stop_run(run_id: str) -> RunSummary:
    try:
        return get_control_state().stop(run_id).summary()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}") from exc


@router.get("/artifacts", response_model=list[ArtifactSummary])
async def artifacts(
    artifact_dir: str = "runtime_artifacts",
    role: str | None = None,
    query: str | None = None,
) -> list[ArtifactSummary]:
    items = _load_artifacts(artifact_dir)
    if role:
        items = [item for item in items if item.role.upper() == role.upper()]
    if query:
        lowered = query.lower()
        items = [
            item
            for item in items
            if lowered in item.content.lower() or lowered in item.task.lower()
        ]
    return items


@router.get("/repository/tree", response_model=RepoTreeNode)
async def repository_tree(
    root: str,
    depth: int = Query(default=3, ge=1, le=8),
) -> RepoTreeNode:
    root_path = _safe_root(root)
    return _build_tree(root_path, root_path, depth)


@router.get("/repository/file")
async def repository_file(root: str, path: str) -> dict[str, object]:
    root_path = _safe_root(root)
    target = (root_path / path).resolve()
    if not target.is_relative_to(root_path) or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return {
        "path": target.relative_to(root_path).as_posix(),
        "content": target.read_text(encoding="utf-8", errors="replace"),
        "size_bytes": target.stat().st_size,
    }


@router.get("/patch")
async def patch(repository_root: str) -> dict[str, object]:
    return _patch_snapshot(repository_root)


@router.get("/logs")
async def logs() -> dict[str, object]:
    return {"logs": _combined_logs(get_control_state())}


@router.get("/events")
async def events() -> StreamingResponse:
    async def stream():
        while True:
            for line in _combined_logs(get_control_state())[-25:]:
                yield f"data: {line}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(stream(), media_type="text/event-stream")


def _load_artifacts(artifact_dir: str) -> list[ArtifactSummary]:
    loader = ArtifactLoader(artifact_dir)
    artifacts: list[ArtifactSummary] = []
    for role in loader.list_roles():
        for artifact in loader.load_role_artifacts(role):
            artifacts.append(
                ArtifactSummary(
                    artifact_id=artifact.artifact_id,
                    role=artifact.role,
                    round_id=artifact.round_id,
                    task=artifact.task,
                    content=artifact.content,
                    created_at=artifact.created_at.isoformat(),
                    metadata=artifact.metadata,
                )
            )
    return sorted(artifacts, key=lambda item: (item.round_id, item.role))


def _courtroom_state(active: RunRecord | None) -> list[dict[str, object]]:
    active_role = "PRIMARY_CODER" if active and active.status == "running" else None
    return [
        {
            "role": role,
            "status": "active" if role == active_role else "waiting",
            "model": (active.request.models or {}).get(role, _default_model_for_role(role)) if active else _default_model_for_role(role),
            "runtime": "vLLM autoswap",
            "token_count": None,
            "inference_time_seconds": None,
            "health": "unknown" if role != active_role else "running",
        }
        for role in COURTROOM_ROLES
    ]


def _timeline_state(active: RunRecord | None) -> list[dict[str, object]]:
    status = active.status if active else "idle"
    completed = set()
    if active and active.result:
        completed = set(TIMELINE_STAGES)
    return [
        {
            "name": stage,
            "status": "completed" if stage in completed else ("active" if status == "running" and stage == "Coder" else "pending"),
        }
        for stage in TIMELINE_STAGES
    ]


def _runtime_snapshot(active: RunRecord | None) -> dict[str, object]:
    return {
        "active_runtime": active.status if active else "idle",
        "active_model": "autoswap-controlled" if active else None,
        "pid": None,
        "pgid": None,
        "memory": _process_memory_snapshot(),
        "vram": _nvidia_smi_snapshot(),
        "swap_count": None,
        "health": "running" if active and active.status == "running" else "idle",
    }


def _patch_snapshot(repo_root: str) -> dict[str, object]:
    try:
        git = GitDiff(repo_root=repo_root)
        return {
            "diff": git.diff_head(),
            "stat": git.stat(),
            "changed_files": git.changed_files(),
            "repository_root": str(Path(repo_root).resolve()),
        }
    except (GitDiffError, OSError, ValueError) as exc:
        return {
            "diff": "",
            "stat": "",
            "changed_files": [],
            "repository_root": repo_root,
            "error": str(exc),
        }


def _tests_snapshot(result: dict | None) -> dict[str, object]:
    if not result:
        return {"status": "idle", "passing": 0, "failing": 0, "retries": 0, "repair_attempts": 0}
    passed = bool(result.get("tests_passed"))
    return {
        "status": "passed" if passed else "failed",
        "passing": 1 if passed else 0,
        "failing": 0 if passed else 1,
        "retries": result.get("iterations_run", 0),
        "repair_attempts": max(int(result.get("iterations_run", 1)) - 1, 0),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }


def _conversation_summaries(artifacts: list[ArtifactSummary]) -> list[dict[str, object]]:
    return [
        {
            "role": artifact.role,
            "round_id": artifact.round_id,
            "summary": artifact.content.strip().splitlines()[0][:240] if artifact.content.strip() else "",
            "created_at": artifact.created_at,
        }
        for artifact in artifacts[-12:]
    ]


def _combined_logs(state: ControlCenterState) -> list[str]:
    lines = state.logs()
    log_dir = Path("runtime_logs")
    if log_dir.exists():
        for path in sorted(log_dir.glob("*.log")):
            try:
                tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            except OSError:
                continue
            lines.extend(f"{path.name}: {line}" for line in tail)
    return lines[-250:]


def _latest_result(state: ControlCenterState) -> dict | None:
    for record in state.runs():
        if record.result:
            return record.result
    return None


def _safe_root(root: str) -> Path:
    root_path = Path(root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise HTTPException(status_code=404, detail=f"repository root not found: {root}")
    return root_path


def _build_tree(root: Path, current: Path, depth: int) -> RepoTreeNode:
    relative = "." if current == root else current.relative_to(root).as_posix()
    node = RepoTreeNode(
        name=current.name or current.as_posix(),
        path=relative,
        type="directory" if current.is_dir() else "file",
        size_bytes=current.stat().st_size if current.is_file() else None,
    )
    if not current.is_dir() or depth <= 0:
        return node

    children: list[RepoTreeNode] = []
    for child in sorted(current.iterdir(), key=lambda path: (path.is_file(), path.name.lower())):
        if child.name in {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}:
            continue
        if child.is_dir() or child.suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".json", ".toml", ".yaml", ".yml", ".css"}:
            children.append(_build_tree(root, child, depth - 1))
        if len(children) >= 200:
            break
    node.children = children
    return node


def _process_memory_snapshot() -> dict[str, object]:
    return {"rss_mb": None, "source": "process-specific runtime metrics unavailable"}


def _nvidia_smi_snapshot() -> dict[str, object]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {"available": False, "error": result.stderr.strip()}
        used, total, utilization = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        return {
            "available": True,
            "memory_used_mb": int(used),
            "memory_total_mb": int(total),
            "gpu_utilization_percent": int(utilization),
        }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}


def _default_model_for_role(role: str) -> str:
    return {
        "PRIMARY_CODER": "qwen-primary",
        "DEEPSEEK_SYNTH": "deepseek-synth",
        "JUDGE": "qwen-judge",
    }[role]
