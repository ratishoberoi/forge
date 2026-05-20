from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from backend.runtime.json_store import atomic_write_json, load_json_store


@dataclass(slots=True)
class ADRRecord:
    adr_id: str
    repository_path: str
    title: str
    status: str = "accepted"
    context: str = ""
    decision: str = ""
    consequences: str = ""
    alternatives_rejected: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "adr_id": self.adr_id,
            "repository_path": self.repository_path,
            "title": self.title,
            "status": self.status,
            "context": self.context,
            "decision": self.decision,
            "consequences": self.consequences,
            "alternatives_rejected": self.alternatives_rejected,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ADRRecord:
        return cls(
            adr_id=str(data["adr_id"]),
            repository_path=str(data["repository_path"]),
            title=str(data.get("title", "")),
            status=str(data.get("status", "accepted")),
            context=str(data.get("context", "")),
            decision=str(data.get("decision", "")),
            consequences=str(data.get("consequences", "")),
            alternatives_rejected=[str(item) for item in data.get("alternatives_rejected", [])],
            tags=[str(item) for item in data.get("tags", [])],
            created_at=str(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=str(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
        )


class ADRStore:
    """Local architectural decision record store."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or ".forge/adrs.json").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        repository_path: str,
        title: str,
        context: str,
        decision: str,
        consequences: str = "",
        alternatives_rejected: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> ADRRecord:
        resolved_repo = str(Path(repository_path).resolve())
        adr_id = _slug(title)
        existing = [record for record in self.list(repository_path=resolved_repo) if record.adr_id == adr_id]
        created_at = existing[0].created_at if existing else datetime.now(timezone.utc).isoformat()
        record = ADRRecord(
            adr_id=adr_id,
            repository_path=resolved_repo,
            title=title,
            context=context,
            decision=decision,
            consequences=consequences,
            alternatives_rejected=alternatives_rejected or [],
            tags=tags or [],
            created_at=created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        records = [item for item in self.list() if not (item.repository_path == resolved_repo and item.adr_id == adr_id)]
        records.append(record)
        self._save(records)
        return record

    def infer_from_frameworks(
        self,
        *,
        repository_path: str,
        frameworks: list[str],
        databases: list[str] | None = None,
    ) -> list[ADRRecord]:
        records: list[ADRRecord] = []
        if frameworks:
            records.append(
                self.create(
                    repository_path=repository_path,
                    title="Use detected application framework conventions",
                    context=f"Repository scan detected frameworks: {', '.join(frameworks)}.",
                    decision="Preserve existing framework conventions during autonomous changes.",
                    consequences="Future plans should avoid unnecessary framework migrations.",
                    alternatives_rejected=["Rewrite application architecture without operator approval."],
                    tags=["framework"],
                )
            )
        if databases:
            records.append(
                self.create(
                    repository_path=repository_path,
                    title="Preserve detected database choices",
                    context=f"Repository scan detected databases: {', '.join(databases)}.",
                    decision="Prefer existing database and migration patterns for new persistence work.",
                    consequences="Schema changes should include tests and migration safety checks.",
                    alternatives_rejected=["Introduce a new database engine by default."],
                    tags=["database"],
                )
            )
        return records

    def list(self, repository_path: str | None = None) -> list[ADRRecord]:
        data = load_json_store(self.path, default={"adrs": []}, store_name="adrs")
        records = [ADRRecord.from_dict(item) for item in data.get("adrs", [])]
        if repository_path:
            resolved = str(Path(repository_path).resolve())
            records = [record for record in records if record.repository_path == resolved]
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def _save(self, records: list[ADRRecord]) -> None:
        atomic_write_json(self.path, {"adrs": [record.to_dict() for record in records]})


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "adr"
