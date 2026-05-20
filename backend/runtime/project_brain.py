from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from backend.runtime.json_store import atomic_write_json, load_json_store


@dataclass(slots=True)
class ProjectBrainRecord:
    repository_path: str
    architecture_summaries: list[str] = field(default_factory=list)
    feature_summaries: list[str] = field(default_factory=list)
    service_boundaries: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    successful_patterns: list[str] = field(default_factory=list)
    previous_objectives: list[str] = field(default_factory=list)
    previous_implementations: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_path": self.repository_path,
            "architecture_summaries": self.architecture_summaries,
            "feature_summaries": self.feature_summaries,
            "service_boundaries": self.service_boundaries,
            "decisions": self.decisions,
            "tradeoffs": self.tradeoffs,
            "failures": self.failures,
            "repairs": self.repairs,
            "successful_patterns": self.successful_patterns,
            "previous_objectives": self.previous_objectives,
            "previous_implementations": self.previous_implementations,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectBrainRecord:
        return cls(
            repository_path=str(data["repository_path"]),
            architecture_summaries=_strings(data.get("architecture_summaries")),
            feature_summaries=_strings(data.get("feature_summaries")),
            service_boundaries=_strings(data.get("service_boundaries")),
            decisions=_strings(data.get("decisions")),
            tradeoffs=_strings(data.get("tradeoffs")),
            failures=_strings(data.get("failures")),
            repairs=_strings(data.get("repairs")),
            successful_patterns=_strings(data.get("successful_patterns")),
            previous_objectives=_strings(data.get("previous_objectives")),
            previous_implementations=_strings(data.get("previous_implementations")),
            updated_at=str(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
        )

    def brief(self, *, objective: str = "") -> dict[str, Any]:
        return {
            "objective": objective,
            "architecture": self.architecture_summaries[-3:],
            "features": self.feature_summaries[-8:],
            "decisions": self.decisions[-8:],
            "tradeoffs": self.tradeoffs[-8:],
            "failures": self.failures[-8:],
            "repairs": self.repairs[-8:],
            "successful_patterns": self.successful_patterns[-8:],
            "previous_objectives": self.previous_objectives[-8:],
            "previous_implementations": self.previous_implementations[-8:],
            "updated_at": self.updated_at,
        }


class ProjectBrain:
    """Local persistent project knowledge that is consulted before planning."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or ".forge/project_brain.json").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, repository_path: str) -> ProjectBrainRecord:
        key = str(Path(repository_path).resolve())
        return self._records().get(key) or ProjectBrainRecord(repository_path=key)

    def update_from_preparation(
        self,
        *,
        repository_path: str,
        objective: str,
        architecture_summary: str,
        service_boundaries: list[str],
        feature_summaries: list[str] | None = None,
    ) -> ProjectBrainRecord:
        key = str(Path(repository_path).resolve())
        record = self.get(key)
        record.architecture_summaries = _dedupe(record.architecture_summaries + [architecture_summary])[-20:]
        record.service_boundaries = _dedupe(record.service_boundaries + service_boundaries)[-80:]
        record.feature_summaries = _dedupe(record.feature_summaries + (feature_summaries or []))[-80:]
        record.previous_objectives = _dedupe(record.previous_objectives + [objective])[-100:]
        record.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_record(record)
        return record

    def record_outcome(
        self,
        *,
        repository_path: str,
        objective: str,
        implementation: str = "",
        failures: list[str] | None = None,
        repairs: list[str] | None = None,
        successful: bool = False,
    ) -> ProjectBrainRecord:
        record = self.get(repository_path)
        record.previous_objectives = _dedupe(record.previous_objectives + [objective])[-100:]
        if implementation:
            record.previous_implementations = _dedupe(record.previous_implementations + [implementation])[-100:]
        record.failures = _dedupe(record.failures + (failures or []))[-100:]
        record.repairs = _dedupe(record.repairs + (repairs or []))[-100:]
        if successful and implementation:
            record.successful_patterns = _dedupe(record.successful_patterns + [implementation])[-80:]
        record.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_record(record)
        return record

    def add_decision(
        self,
        *,
        repository_path: str,
        decision: str,
        tradeoff: str = "",
    ) -> ProjectBrainRecord:
        record = self.get(repository_path)
        record.decisions = _dedupe(record.decisions + [decision])[-100:]
        if tradeoff:
            record.tradeoffs = _dedupe(record.tradeoffs + [tradeoff])[-100:]
        record.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_record(record)
        return record

    def list_records(self) -> list[ProjectBrainRecord]:
        return sorted(self._records().values(), key=lambda record: record.updated_at, reverse=True)

    def _save_record(self, record: ProjectBrainRecord) -> None:
        records = self._records()
        records[record.repository_path] = record
        payload = {"repositories": {key: value.to_dict() for key, value in records.items()}}
        atomic_write_json(self.path, payload)

    def _records(self) -> dict[str, ProjectBrainRecord]:
        data = load_json_store(
            self.path,
            default={"repositories": {}},
            store_name="project_brain",
        )
        return {
            str(key): ProjectBrainRecord.from_dict(value)
            for key, value in dict(data.get("repositories", {})).items()
        }


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
