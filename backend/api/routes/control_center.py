"""Operator dashboard routes for Forge Control Center."""

from __future__ import annotations

import os
import subprocess
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
from backend.runtime.architecture_memory import ArchitectureMemory
from backend.runtime.autonomous_run import AutonomousRun
from backend.runtime.benchmark_suite import BenchmarkCase, BenchmarkSuite
from backend.runtime.git_diff import GitDiff, GitDiffError
from backend.runtime.git_manager import GitManager, GitManagerError
from backend.runtime.repository_execution_engine import (
    RepositoryExecutionEngine,
    RepositoryExecutionError,
)
from backend.runtime.run_history import RunHistoryError, RunHistoryStore
from backend.runtime.long_horizon import ObjectiveMemory
from backend.runtime.workspace_manager import WorkspaceManager, WorkspaceManagerError


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
    repository_id: str | None = None
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
    repository_id: str | None = None
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
    convergence: dict[str, object]
    logs: list[str]
    conversation: list[dict[str, object]]
    repository_summary: dict[str, object] | None = None
    architecture_summary: str | None = None
    execution_plan: dict[str, object] | None = None
    repositories: list[dict[str, object]] = Field(default_factory=list)
    active_repository: dict[str, object] | None = None
    git: dict[str, object] | None = None
    run_history: list[dict[str, object]] = Field(default_factory=list)
    queued_tasks: list[dict[str, object]] = Field(default_factory=list)
    architecture_memory: dict[str, object] | None = None
    task_plan: dict[str, object] | None = None
    execution_graph: dict[str, object] | None = None
    compressed_context: dict[str, object] | None = None
    objective_memory: list[dict[str, object]] = Field(default_factory=list)
    bootstrap: dict[str, object] | None = None
    acceptance: dict[str, object] | None = None
    build_validation: dict[str, object] | None = None
    visual_validation: dict[str, object] | None = None
    quality_score: dict[str, object] | None = None
    release_report: dict[str, object] | None = None


class ImportRepositoryRequest(BaseModel):
    path: str
    repository_name: str | None = None
    set_active: bool = True


class CloneRepositoryRequest(BaseModel):
    source: str
    repository_name: str | None = None
    set_active: bool = True


class BranchRequest(BaseModel):
    repository_root: str | None = None
    repository_id: str | None = None
    branch: str


class CommitRequest(BaseModel):
    repository_root: str | None = None
    repository_id: str | None = None
    message: str


class RollbackRequest(BaseModel):
    repository_root: str | None = None
    repository_id: str | None = None
    target: str = "HEAD"
    clean_untracked: bool = False


class RevertRequest(BaseModel):
    repository_root: str | None = None
    repository_id: str | None = None
    commit_sha: str
    no_commit: bool = False


class BenchmarkRequest(BaseModel):
    root: str = ".forge/benchmarks"
    cleanup: bool = True
    cases: list[dict[str, object]] = Field(default_factory=list)


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
            repository_id=self.request.repository_id,
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
        self._execution_lock = Lock()
        self.workspace_manager = WorkspaceManager()
        self.run_history = RunHistoryStore()

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
        with self._execution_lock:
            self._execute_run_locked(run_id)

    def _execute_run_locked(self, run_id: str) -> None:
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
        branch: str | None = None
        try:
            self.run_history.record_started(
                run_id=record.id,
                objective=record.request.objective,
                repository_id=record.request.repository_id,
                repository_path=record.request.repository_root,
            )
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
                run_id=record.id,
            )
            record.result = result
            branch = result.get("execution_branch") if isinstance(result, dict) else None
            record.status = "completed"
            self.run_history.record_completed(
                run_id=record.id,
                status=record.status,
                result=result,
                telemetry=_telemetry_from_result(result),
                branch=branch,
            )
            self.log(f"[RUN] completed {run_id}")
        except Exception as exc:
            record.error = str(exc)
            record.status = "failed"
            try:
                self.run_history.record_completed(
                    run_id=record.id,
                    status=record.status,
                    result=record.result,
                    telemetry=[f"[RUN] failed {record.id}: {exc}"],
                    branch=branch,
                )
            except RunHistoryError:
                pass
            self.log(f"[RUN] failed {run_id}: {exc}")
        finally:
            record.completed_at = datetime.now(timezone.utc)
            record.touch()


router = APIRouter(prefix="/api/control", tags=["control-center"])


def get_control_state() -> ControlCenterState:
    if not hasattr(router, "_control_state"):
        router._control_state = ControlCenterState()  # type: ignore[attr-defined]
    return router._control_state  # type: ignore[attr-defined]


@router.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "backend": "forge-control-center",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


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
    repositories = [record.to_dict() for record in state.workspace_manager.list_repositories()]
    active_repository = (
        state.workspace_manager.active_repository().to_dict()
        if state.workspace_manager.active_repository()
        else None
    )
    preparation = await _repository_preparation(
        repo_root,
        active.request.objective if active else "Inspect repository",
    )
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
        convergence=_convergence_snapshot(latest_result),
        logs=_combined_logs(state, latest_result),
        conversation=_conversation_summaries(artifacts),
        repository_summary=preparation["scan"] if preparation else None,
        architecture_summary=(
            preparation["scan"].get("architecture_summary")
            if preparation and isinstance(preparation.get("scan"), dict)
            else None
        ),
        execution_plan=preparation["plan"] if preparation else None,
        repositories=repositories,
        active_repository=active_repository,
        git=_git_snapshot(repo_root),
        run_history=[record.to_dict() for record in state.run_history.list_runs()[:25]],
        queued_tasks=[record.summary().model_dump() for record in state.runs() if record.status == "queued"],
        architecture_memory=(
            preparation.get("architecture_memory")
            if preparation and isinstance(preparation.get("architecture_memory"), dict)
            else _architecture_memory_snapshot(repo_root)
        ),
        task_plan=preparation.get("task_plan") if preparation else None,
        execution_graph=preparation.get("execution_graph") if preparation else None,
        compressed_context=preparation.get("compressed_context") if preparation else None,
        objective_memory=[
            record.to_dict()
            for record in ObjectiveMemory().list_records(repository_path=repo_root)[:20]
        ],
        bootstrap=_result_section(latest_result, "bootstrap"),
        acceptance=_result_section(latest_result, "acceptance"),
        build_validation=_result_section(latest_result, "build_validation"),
        visual_validation=_result_section(latest_result, "visual_validation"),
        quality_score=_result_section(latest_result, "quality_score"),
        release_report=_result_section(latest_result, "release_report"),
    )


@router.post("/runs", response_model=RunSummary)
async def create_run(request: RunRequest, background_tasks: BackgroundTasks) -> RunSummary:
    state = get_control_state()
    if request.repository_id:
        try:
            repository = state.workspace_manager.get_repository(request.repository_id)
            request = request.model_copy(update={"repository_root": repository.repository_path})
        except WorkspaceManagerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    repo_root = Path(request.repository_root).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise HTTPException(status_code=400, detail=f"repository_root does not exist: {repo_root}")
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
    state = get_control_state()
    return {"logs": _combined_logs(state, _latest_result(state))}


@router.get("/events")
async def events() -> StreamingResponse:
    async def stream():
        while True:
            for line in _combined_logs(get_control_state())[-25:]:
                yield f"data: {line}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/workspaces/repositories")
async def list_repositories(include_archived: bool = False) -> dict[str, object]:
    manager = get_control_state().workspace_manager
    active = manager.active_repository()
    return {
        "repositories": [
            record.to_dict()
            for record in manager.list_repositories(include_archived=include_archived)
        ],
        "active_repository": active.to_dict() if active else None,
    }


@router.post("/workspaces/import")
async def import_repository(request: ImportRepositoryRequest) -> dict[str, object]:
    try:
        record = get_control_state().workspace_manager.import_local_repository(
            request.path,
            repository_name=request.repository_name,
            set_active=request.set_active,
        )
        return record.to_dict()
    except WorkspaceManagerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/workspaces/clone")
async def clone_repository(request: CloneRepositoryRequest) -> dict[str, object]:
    try:
        record = get_control_state().workspace_manager.clone_repository(
            request.source,
            repository_name=request.repository_name,
            set_active=request.set_active,
        )
        return record.to_dict()
    except WorkspaceManagerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/workspaces/repositories/{repository_id}/switch")
async def switch_repository(repository_id: str) -> dict[str, object]:
    try:
        return get_control_state().workspace_manager.switch_active_repository(repository_id).to_dict()
    except WorkspaceManagerError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/workspaces/repositories/{repository_id}/archive")
async def archive_repository(repository_id: str) -> dict[str, object]:
    try:
        return get_control_state().workspace_manager.archive_repository(repository_id).to_dict()
    except WorkspaceManagerError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/workspaces/repositories/{repository_id}")
async def delete_repository_metadata(repository_id: str) -> dict[str, str]:
    get_control_state().workspace_manager.delete_repository_metadata(repository_id)
    return {"status": "deleted", "repository_id": repository_id}


@router.post("/workspaces/repositories/{repository_id}/refresh")
async def refresh_repository(repository_id: str, force: bool = False) -> dict[str, object]:
    try:
        record = await get_control_state().workspace_manager.refresh_intelligence(
            repository_id,
            force=force,
        )
        return record.to_dict()
    except WorkspaceManagerError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/git/status")
async def git_status(
    repository_root: str | None = None,
    repository_id: str | None = None,
) -> dict[str, object]:
    return _git_manager(repository_root=repository_root, repository_id=repository_id).status().to_dict()


@router.get("/git/branches")
async def git_branches(
    repository_root: str | None = None,
    repository_id: str | None = None,
) -> dict[str, object]:
    manager = _git_manager(repository_root=repository_root, repository_id=repository_id)
    return {"current": manager.current_branch(), "branches": manager.branches()}


@router.post("/git/branches")
async def create_branch(request: BranchRequest) -> dict[str, object]:
    try:
        manager = _git_manager(repository_root=request.repository_root, repository_id=request.repository_id)
        branch = manager.create_branch(request.branch)
        return {"branch": branch, "status": manager.status().to_dict()}
    except GitManagerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/git/history")
async def git_history(
    repository_root: str | None = None,
    repository_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, object]:
    manager = _git_manager(repository_root=repository_root, repository_id=repository_id)
    return {"commits": [commit.to_dict() for commit in manager.history(limit=limit)]}


@router.post("/git/commit")
async def git_commit(request: CommitRequest) -> dict[str, object]:
    try:
        manager = _git_manager(repository_root=request.repository_root, repository_id=request.repository_id)
        sha = manager.commit_all(request.message)
        return {"commit_sha": sha, "status": manager.status().to_dict()}
    except GitManagerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/git/rollback")
async def git_rollback(request: RollbackRequest) -> dict[str, object]:
    try:
        manager = _git_manager(repository_root=request.repository_root, repository_id=request.repository_id)
        manager.rollback(request.target, clean_untracked=request.clean_untracked)
        return {"status": manager.status().to_dict()}
    except GitManagerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/git/revert")
async def git_revert(request: RevertRequest) -> dict[str, object]:
    try:
        manager = _git_manager(repository_root=request.repository_root, repository_id=request.repository_id)
        sha = manager.revert(request.commit_sha, no_commit=request.no_commit)
        return {"commit_sha": sha, "status": manager.status().to_dict()}
    except GitManagerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/history/runs")
async def history_runs(repository_id: str | None = None) -> dict[str, object]:
    records = get_control_state().run_history.list_runs(repository_id=repository_id)
    return {"runs": [record.to_dict() for record in records]}


@router.get("/history/runs/{run_id}")
async def history_run(run_id: str) -> dict[str, object]:
    try:
        return get_control_state().run_history.get(run_id).to_dict()
    except RunHistoryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/history/runs/{run_id}/replay")
async def history_replay(run_id: str) -> dict[str, object]:
    try:
        return {"run_id": run_id, "events": get_control_state().run_history.replay(run_id)}
    except RunHistoryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/memory/architecture")
async def architecture_memory(repository_root: str | None = None) -> dict[str, object]:
    if repository_root:
        return {"memory": _architecture_memory_snapshot(repository_root)}
    return {"repositories": [record.to_dict() for record in ArchitectureMemory().list_records()]}


@router.get("/memory/objectives")
async def objective_memory(repository_root: str | None = None) -> dict[str, object]:
    return {
        "objectives": [
            record.to_dict()
            for record in ObjectiveMemory().list_records(repository_path=repository_root)
        ]
    }


@router.post("/benchmarks/run")
async def run_benchmarks(request: BenchmarkRequest) -> dict[str, object]:
    cases = [
        BenchmarkCase(
            name=str(item.get("name", f"case-{index}")),
            objective=str(item.get("objective", "")),
            expected_files=[str(path) for path in item.get("expected_files", [])]
            if isinstance(item.get("expected_files"), list)
            else [],
        )
        for index, item in enumerate(request.cases, start=1)
        if str(item.get("objective", "")).strip()
    ]
    results = BenchmarkSuite(root=request.root).run(
        cases=cases or None,
        cleanup=request.cleanup,
    )
    total = len(results)
    successes = sum(1 for result in results if result.success)
    return {
        "results": [result.to_dict() for result in results],
        "summary": {
            "total": total,
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "completion_rate": sum(result.completion_rate for result in results) / total if total else 0.0,
            "convergence_rate": sum(result.convergence_rate for result in results) / total if total else 0.0,
            "repair_rate": sum(result.repair_rate for result in results) / total if total else 0.0,
        },
    }


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


def _git_snapshot(repo_root: str) -> dict[str, object]:
    try:
        manager = GitManager(repo_root)
        return {
            "status": manager.status().to_dict(),
            "branches": manager.branches(),
            "history": [commit.to_dict() for commit in manager.history(limit=10)],
            "changed_files": manager.diff_name_status(),
        }
    except GitManagerError as exc:
        return {"error": str(exc), "status": None, "branches": [], "history": [], "changed_files": []}


def _architecture_memory_snapshot(repo_root: str) -> dict[str, object] | None:
    record = ArchitectureMemory().get(repo_root)
    return record.to_dict() if record else None


def _tests_snapshot(result: dict | None) -> dict[str, object]:
    if not result:
        return {"status": "idle", "passing": 0, "failing": 0, "retries": 0, "repair_attempts": 0}
    repair = result.get("repair_convergence")
    if isinstance(repair, dict):
        state = repair.get("state") if isinstance(repair.get("state"), dict) else {}
        execution = repair.get("final_execution") if isinstance(repair.get("final_execution"), dict) else {}
        passed = bool(execution.get("passed"))
        return {
            "status": "passed" if passed else "failed",
            "passing": 1 if passed else 0,
            "failing": 0 if passed else 1,
            "retries": state.get("iteration_count", 0),
            "repair_attempts": state.get("repair_count", 0),
            "stdout": execution.get("stdout", ""),
            "stderr": execution.get("stderr", ""),
            "pass_rate": state.get("test_pass_rate", 0.0),
            "last_failure_type": state.get("last_failure_type"),
            "last_failing_test": state.get("last_failing_test"),
        }
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


def _convergence_snapshot(result: dict | None) -> dict[str, object]:
    if not result:
        return {
            "status": "idle",
            "current_phase": "idle",
            "current_repair_attempt": 0,
            "repair_limit": 0,
            "success": False,
            "stop_reason": None,
            "failure_category": None,
            "last_failing_test": None,
            "test_pass_rate": 0.0,
            "timeline": [],
            "history": [],
        }

    repair = result.get("repair_convergence")
    if not isinstance(repair, dict):
        return {
            "status": "unknown",
            "current_phase": "legacy",
            "current_repair_attempt": result.get("repair_attempts", 0),
            "repair_limit": 0,
            "success": bool(result.get("tests_passed")),
            "stop_reason": result.get("stop_reason"),
            "failure_category": result.get("failure_type"),
            "last_failing_test": None,
            "test_pass_rate": 1.0 if result.get("tests_passed") else 0.0,
            "timeline": [],
            "history": [],
        }

    state = repair.get("state") if isinstance(repair.get("state"), dict) else {}
    history = state.get("history") if isinstance(state.get("history"), list) else []
    return {
        "status": state.get("status", "unknown"),
        "current_phase": state.get("current_phase", "unknown"),
        "current_repair_attempt": state.get("repair_count", 0),
        "repair_limit": state.get("repair_limit", 0),
        "success": bool(state.get("success")),
        "stop_reason": state.get("stop_reason"),
        "failure_category": state.get("last_failure_type"),
        "last_failing_test": state.get("last_failing_test"),
        "test_pass_rate": state.get("test_pass_rate", 0.0),
        "timeline": history,
        "history": history,
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


def _combined_logs(state: ControlCenterState, latest_result: dict | None = None) -> list[str]:
    lines = state.logs()
    latest_result = latest_result if latest_result is not None else _latest_result(state)
    if latest_result:
        repair = latest_result.get("repair_convergence")
        telemetry = repair.get("telemetry") if isinstance(repair, dict) else None
        if isinstance(telemetry, list):
            lines.extend(str(line) for line in telemetry)
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


def _result_section(result: dict | None, key: str) -> dict[str, object] | None:
    value = result.get(key) if isinstance(result, dict) else None
    return value if isinstance(value, dict) else None


def _telemetry_from_result(result: dict | None) -> list[str]:
    if not result:
        return []
    repair = result.get("repair_convergence")
    telemetry = repair.get("telemetry") if isinstance(repair, dict) else None
    return [str(line) for line in telemetry] if isinstance(telemetry, list) else []


def _git_manager(
    *,
    repository_root: str | None = None,
    repository_id: str | None = None,
) -> GitManager:
    if repository_id:
        try:
            repository = get_control_state().workspace_manager.get_repository(repository_id)
            return GitManager(repository.repository_path)
        except (WorkspaceManagerError, GitManagerError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    if repository_root:
        try:
            return GitManager(repository_root)
        except GitManagerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    active = get_control_state().workspace_manager.active_repository()
    if active:
        try:
            return GitManager(active.repository_path)
        except GitManagerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail="repository_root or repository_id is required")


async def _repository_preparation(repo_root: str, objective: str) -> dict[str, object] | None:
    try:
        engine = RepositoryExecutionEngine(repo_root=repo_root)
        return (await engine.prepare(objective)).to_dict()
    except (RepositoryExecutionError, OSError, ValueError):
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
