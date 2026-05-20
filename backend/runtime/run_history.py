from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunHistoryError(RuntimeError):
    """Raised when run history persistence fails."""


@dataclass(slots=True)
class RunHistoryRecord:
    run_id: str
    objective: str
    repository_id: str | None
    repository_path: str
    status: str
    created_at: str
    updated_at: str
    branch: str | None = None
    commit_sha: str | None = None
    result: dict[str, Any] | None = None
    telemetry: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "repository_id": self.repository_id,
            "repository_path": self.repository_path,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "result": self.result,
            "telemetry": self.telemetry,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunHistoryRecord:
        return cls(
            run_id=str(data["run_id"]),
            objective=str(data["objective"]),
            repository_id=data.get("repository_id"),
            repository_path=str(data["repository_path"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            branch=data.get("branch"),
            commit_sha=data.get("commit_sha"),
            result=data.get("result"),
            telemetry=[str(item) for item in data.get("telemetry", [])],
        )


class RunHistoryStore:
    def __init__(self, path: str | None = None) -> None:
        default_path = ".forge/test_run_history.json" if _running_under_pytest() else ".forge/run_history.json"
        self._filter_pytest_records = path is None and not _running_under_pytest()
        self.path = Path(path or default_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list_runs(self, *, repository_id: str | None = None) -> list[RunHistoryRecord]:
        records = self._load()
        if self._filter_pytest_records:
            records = [record for record in records if not _is_pytest_record(record)]
        if repository_id:
            records = [record for record in records if record.repository_id == repository_id]
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def get(self, run_id: str) -> RunHistoryRecord:
        for record in self._load():
            if record.run_id == run_id:
                return record
        raise RunHistoryError(f"run history not found: {run_id}")

    def upsert(self, record: RunHistoryRecord) -> RunHistoryRecord:
        records = [item for item in self._load() if item.run_id != record.run_id]
        record.updated_at = _now()
        records.append(record)
        self._save(records)
        return record

    def record_started(
        self,
        *,
        run_id: str,
        objective: str,
        repository_path: str,
        repository_id: str | None = None,
        branch: str | None = None,
    ) -> RunHistoryRecord:
        now = _now()
        record = RunHistoryRecord(
            run_id=run_id,
            objective=objective,
            repository_id=repository_id,
            repository_path=repository_path,
            status="running",
            created_at=now,
            updated_at=now,
            branch=branch,
        )
        return self.upsert(record)

    def record_completed(
        self,
        *,
        run_id: str,
        status: str,
        result: dict[str, Any] | None,
        telemetry: list[str],
        branch: str | None = None,
        commit_sha: str | None = None,
    ) -> RunHistoryRecord:
        record = self.get(run_id)
        record.status = status
        record.result = result
        record.telemetry = telemetry
        record.branch = branch or record.branch
        record.commit_sha = commit_sha
        return self.upsert(record)

    def replay(self, run_id: str) -> list[dict[str, Any]]:
        record = self.get(run_id)
        telemetry = record.telemetry
        if not telemetry and record.result:
            repair = record.result.get("repair_convergence")
            telemetry = repair.get("telemetry", []) if isinstance(repair, dict) else []
        stages = [
            "REPOSITORY_SCAN",
            "PLANNING",
            "CODER",
            "SYNTH",
            "JUDGE",
            "REPAIR",
            "CONVERGENCE",
        ]
        events = [{"stage": stage, "status": "completed", "message": stage} for stage in stages]
        events.extend(
            {"stage": _stage_from_line(line), "status": "event", "message": line}
            for line in telemetry
        )
        return events

    def _load(self) -> list[RunHistoryRecord]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RunHistoryError(f"failed to read run history: {exc}") from exc
        return [RunHistoryRecord.from_dict(item) for item in data.get("runs", [])]

    def _save(self, records: list[RunHistoryRecord]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps({"runs": [record.to_dict() for record in records]}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)


def _stage_from_line(line: str) -> str:
    if "[REPAIR" in line:
        return "REPAIR"
    if "[CONVERGED]" in line:
        return "CONVERGENCE"
    if "[TEST" in line:
        return "TEST_EXECUTION"
    if "[PATCH" in line:
        return "PATCH_APPLY"
    if "[SWAP]" in line or "[INFER]" in line:
        return "COURTROOM"
    return "TELEMETRY"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _running_under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def _is_pytest_record(record: RunHistoryRecord) -> bool:
    path = record.repository_path.replace("\\", "/")
    return "/pytest-of-" in path or "/pytest-" in path or path.startswith("/tmp/pytest")
