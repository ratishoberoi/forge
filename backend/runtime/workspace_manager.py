from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from backend.config.settings import Settings, get_settings
from backend.runtime.git_manager import GitManager, GitManagerError
from backend.runtime.repository_execution_engine import RepositoryExecutionEngine


RepositoryType = Literal["local", "git", "github"]


class WorkspaceManagerError(RuntimeError):
    """Raised when workspace registry operations fail."""


@dataclass(slots=True)
class RepositoryRecord:
    workspace_id: str
    repository_id: str
    repository_name: str
    repository_path: str
    repository_type: RepositoryType
    created_at: str
    last_used: str
    archived: bool = False
    active: bool = False
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    intelligence: dict[str, Any] | None = None
    intelligence_signature: str | None = None
    intelligence_cached_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "repository_id": self.repository_id,
            "repository_name": self.repository_name,
            "repository_path": self.repository_path,
            "repository_type": self.repository_type,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "archived": self.archived,
            "active": self.active,
            "source": self.source,
            "metadata": self.metadata,
            "intelligence": self.intelligence,
            "intelligence_signature": self.intelligence_signature,
            "intelligence_cached_at": self.intelligence_cached_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepositoryRecord:
        return cls(
            workspace_id=str(data["workspace_id"]),
            repository_id=str(data["repository_id"]),
            repository_name=str(data["repository_name"]),
            repository_path=str(data["repository_path"]),
            repository_type=data.get("repository_type", "local"),
            created_at=str(data["created_at"]),
            last_used=str(data["last_used"]),
            archived=bool(data.get("archived", False)),
            active=bool(data.get("active", False)),
            source=data.get("source"),
            metadata=dict(data.get("metadata") or {}),
            intelligence=data.get("intelligence"),
            intelligence_signature=data.get("intelligence_signature"),
            intelligence_cached_at=data.get("intelligence_cached_at"),
        )


class WorkspaceManager:
    def __init__(
        self,
        *,
        registry_path: str | None = None,
        workspace_root: str | None = None,
        settings: Settings | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry_path = Path(registry_path or ".forge/workspace_registry.json").resolve()
        self.workspace_root = Path(workspace_root or ".forge/workspaces").resolve()
        self.env = {**os.environ, **(env or {})}
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def list_repositories(self, *, include_archived: bool = False) -> list[RepositoryRecord]:
        records = self._load()
        if not include_archived:
            records = [record for record in records if not record.archived]
        return sorted(records, key=lambda record: record.last_used, reverse=True)

    def get_repository(self, repository_id: str) -> RepositoryRecord:
        for record in self._load():
            if record.repository_id == repository_id:
                return record
        raise WorkspaceManagerError(f"repository not found: {repository_id}")

    def active_repository(self) -> RepositoryRecord | None:
        for record in self._load():
            if record.active and not record.archived:
                return record
        return None

    def import_local_repository(
        self,
        path: str,
        *,
        repository_name: str | None = None,
        set_active: bool = True,
    ) -> RepositoryRecord:
        repo_path = Path(path).resolve()
        if not repo_path.exists() or not repo_path.is_dir():
            raise WorkspaceManagerError(f"repository path does not exist: {repo_path}")
        record = self._record(
            repository_name=repository_name or repo_path.name,
            repository_path=repo_path,
            repository_type="local",
            source=str(repo_path),
            active=set_active,
        )
        return self._upsert(record, set_active=set_active)

    def clone_repository(
        self,
        source: str,
        *,
        repository_name: str | None = None,
        set_active: bool = True,
    ) -> RepositoryRecord:
        clone_url, repository_type = _normalize_clone_source(source)
        name = repository_name or _repository_name_from_source(source)
        destination = self.workspace_root / f"{name}-{uuid.uuid4().hex[:8]}"
        self._run("git", "clone", clone_url, str(destination), cwd=self.workspace_root)
        record = self._record(
            repository_name=name,
            repository_path=destination,
            repository_type=repository_type,
            source=source,
            active=set_active,
        )
        return self._upsert(record, set_active=set_active)

    def switch_active_repository(self, repository_id: str) -> RepositoryRecord:
        records = self._load()
        selected: RepositoryRecord | None = None
        now = _now()
        for record in records:
            record.active = record.repository_id == repository_id and not record.archived
            if record.active:
                record.last_used = now
                selected = record
        if selected is None:
            raise WorkspaceManagerError(f"repository not found or archived: {repository_id}")
        self._save(records)
        return selected

    def archive_repository(self, repository_id: str) -> RepositoryRecord:
        records = self._load()
        for record in records:
            if record.repository_id == repository_id:
                record.archived = True
                record.active = False
                record.last_used = _now()
                self._save(records)
                return record
        raise WorkspaceManagerError(f"repository not found: {repository_id}")

    def delete_repository_metadata(self, repository_id: str) -> None:
        records = [record for record in self._load() if record.repository_id != repository_id]
        self._save(records)

    async def refresh_intelligence(
        self,
        repository_id: str,
        *,
        objective: str = "Inspect repository",
        force: bool = False,
    ) -> RepositoryRecord:
        record = self.get_repository(repository_id)
        signature = self._signature(record)
        if record.intelligence and record.intelligence_signature == signature and not force:
            return record

        engine = RepositoryExecutionEngine(
            repo_root=record.repository_path,
            settings=self.settings,
        )
        preparation = await engine.prepare(objective)
        record.intelligence = preparation.to_dict()
        record.intelligence_signature = signature
        record.intelligence_cached_at = _now()
        record.metadata = {
            **record.metadata,
            "language": preparation.scan.primary_language,
            "frameworks": preparation.scan.frameworks,
            "package_managers": preparation.scan.package_managers,
            "test_frameworks": preparation.scan.test_frameworks,
            "architecture_summary": preparation.scan.architecture_summary,
            "branch": self._branch(record),
        }
        self._replace(record)
        return record

    def repository_git_status(self, repository_id: str) -> dict[str, Any]:
        record = self.get_repository(repository_id)
        manager = GitManager(record.repository_path, env=self.env)
        return manager.status().to_dict()

    def _record(
        self,
        *,
        repository_name: str,
        repository_path: Path,
        repository_type: RepositoryType,
        source: str,
        active: bool,
    ) -> RepositoryRecord:
        now = _now()
        return RepositoryRecord(
            workspace_id=f"ws-{uuid.uuid4().hex}",
            repository_id=f"repo-{uuid.uuid4().hex}",
            repository_name=repository_name,
            repository_path=str(repository_path),
            repository_type=repository_type,
            created_at=now,
            last_used=now,
            active=active,
            source=source,
            metadata={"branch": self._branch_path(repository_path)},
        )

    def _upsert(self, record: RepositoryRecord, *, set_active: bool) -> RepositoryRecord:
        records = [
            existing
            for existing in self._load()
            if Path(existing.repository_path).resolve() != Path(record.repository_path).resolve()
        ]
        if set_active:
            for existing in records:
                existing.active = False
        records.append(record)
        self._save(records)
        return record

    def _replace(self, record: RepositoryRecord) -> None:
        records = [record if item.repository_id == record.repository_id else item for item in self._load()]
        self._save(records)

    def _load(self) -> list[RepositoryRecord]:
        if not self.registry_path.exists():
            return []
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise WorkspaceManagerError(f"failed to read workspace registry: {exc}") from exc
        return [RepositoryRecord.from_dict(item) for item in data.get("repositories", [])]

    def _save(self, records: list[RepositoryRecord]) -> None:
        payload = {"repositories": [record.to_dict() for record in records]}
        tmp_path = self.registry_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.registry_path)

    def _signature(self, record: RepositoryRecord) -> str:
        try:
            return GitManager(record.repository_path, env=self.env).head_sha()
        except GitManagerError:
            root = Path(record.repository_path)
            latest = max((path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()), default=0)
            return str(latest)

    def _branch(self, record: RepositoryRecord) -> str | None:
        return self._branch_path(Path(record.repository_path))

    def _branch_path(self, path: Path) -> str | None:
        try:
            return GitManager(str(path), env=self.env).current_branch()
        except GitManagerError:
            return None

    def _run(self, *args: str, cwd: Path) -> str:
        try:
            result = subprocess.run(
                list(args),
                cwd=cwd,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise WorkspaceManagerError(f"{' '.join(args)} failed: {exc}") from exc
        if result.returncode != 0:
            if args[:2] == ("git", "clone") and Path(args[-1]).exists():
                shutil.rmtree(args[-1], ignore_errors=True)
            raise WorkspaceManagerError(result.stderr.strip() or f"{' '.join(args)} failed")
        return result.stdout.strip()


def _normalize_clone_source(source: str) -> tuple[str, RepositoryType]:
    trimmed = source.strip()
    if not trimmed:
        raise WorkspaceManagerError("clone source must not be blank.")
    if re_match_github_slug(trimmed):
        return f"https://github.com/{trimmed}.git", "github"
    if "github.com" in trimmed:
        return trimmed, "github"
    return trimmed, "git"


def re_match_github_slug(value: str) -> bool:
    parts = value.split("/")
    return len(parts) == 2 and all(parts) and not value.startswith(("http:", "https:", "git@"))


def _repository_name_from_source(source: str) -> str:
    value = source.rstrip("/").removesuffix(".git")
    return Path(value).name or "repository"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
