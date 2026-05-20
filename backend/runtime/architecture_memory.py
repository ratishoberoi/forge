from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ArchitectureMemoryRecord:
    repository_path: str
    architecture_summary: str = ""
    important_modules: list[str] = field(default_factory=list)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    entrypoints: list[str] = field(default_factory=list)
    service_boundaries: list[str] = field(default_factory=list)
    previously_modified_files: list[str] = field(default_factory=list)
    recurring_failure_patterns: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_path": self.repository_path,
            "architecture_summary": self.architecture_summary,
            "important_modules": self.important_modules,
            "dependency_graph": self.dependency_graph,
            "entrypoints": self.entrypoints,
            "service_boundaries": self.service_boundaries,
            "previously_modified_files": self.previously_modified_files,
            "recurring_failure_patterns": self.recurring_failure_patterns,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureMemoryRecord:
        return cls(
            repository_path=str(data["repository_path"]),
            architecture_summary=str(data.get("architecture_summary", "")),
            important_modules=[str(item) for item in data.get("important_modules", [])],
            dependency_graph={
                str(key): [str(item) for item in value]
                for key, value in dict(data.get("dependency_graph", {})).items()
            },
            entrypoints=[str(item) for item in data.get("entrypoints", [])],
            service_boundaries=[str(item) for item in data.get("service_boundaries", [])],
            previously_modified_files=[str(item) for item in data.get("previously_modified_files", [])],
            recurring_failure_patterns=[str(item) for item in data.get("recurring_failure_patterns", [])],
            updated_at=str(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
        )


class ArchitectureMemory:
    """Persistent repository architecture memory keyed by repository root."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or ".forge/architecture_memory.json").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, repository_path: str) -> ArchitectureMemoryRecord | None:
        key = str(Path(repository_path).resolve())
        return self._records().get(key)

    def upsert_from_preparation(
        self,
        *,
        repository_path: str,
        scan: Any,
        context: Any,
        modified_files: list[str] | None = None,
        failure_patterns: list[str] | None = None,
    ) -> ArchitectureMemoryRecord:
        key = str(Path(repository_path).resolve())
        existing = self.get(key) or ArchitectureMemoryRecord(repository_path=key)
        dependency_graph = _merge_graphs(
            existing.dependency_graph,
            build_dependency_graph(context.imports, context.file_summaries.keys()),
        )
        record = ArchitectureMemoryRecord(
            repository_path=key,
            architecture_summary=scan.architecture_summary,
            important_modules=_dedupe(
                list(existing.important_modules)
                + list(scan.entrypoints)
                + list(context.relevant_files)
                + list(context.related_tests)
            )[:40],
            dependency_graph=dependency_graph,
            entrypoints=_dedupe(list(existing.entrypoints) + list(scan.entrypoints)),
            service_boundaries=_infer_service_boundaries(context.file_summaries.keys()),
            previously_modified_files=_dedupe(
                list(existing.previously_modified_files) + list(modified_files or [])
            )[-100:],
            recurring_failure_patterns=_dedupe(
                list(existing.recurring_failure_patterns) + list(failure_patterns or [])
            )[-50:],
        )
        records = self._records()
        records[key] = record
        self._save(records)
        return record

    def record_outcome(
        self,
        *,
        repository_path: str,
        modified_files: list[str],
        failure_patterns: list[str],
    ) -> ArchitectureMemoryRecord:
        key = str(Path(repository_path).resolve())
        existing = self.get(key) or ArchitectureMemoryRecord(repository_path=key)
        existing.previously_modified_files = _dedupe(
            list(existing.previously_modified_files) + modified_files
        )[-100:]
        existing.recurring_failure_patterns = _dedupe(
            list(existing.recurring_failure_patterns) + failure_patterns
        )[-50:]
        existing.updated_at = datetime.now(timezone.utc).isoformat()
        records = self._records()
        records[key] = existing
        self._save(records)
        return existing

    def list_records(self) -> list[ArchitectureMemoryRecord]:
        return sorted(self._records().values(), key=lambda record: record.updated_at, reverse=True)

    def _records(self) -> dict[str, ArchitectureMemoryRecord]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            key: ArchitectureMemoryRecord.from_dict(value)
            for key, value in data.get("repositories", {}).items()
        }

    def _save(self, records: dict[str, ArchitectureMemoryRecord]) -> None:
        payload = {"repositories": {key: record.to_dict() for key, record in records.items()}}
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)


def build_dependency_graph(
    imports: dict[str, list[str]],
    known_files: Any,
) -> dict[str, list[str]]:
    known = [str(path) for path in known_files]
    graph: dict[str, list[str]] = {}
    module_to_path = {_module_name(path): path for path in known}
    for path, lines in imports.items():
        neighbors: list[str] = []
        for line in lines:
            module = _import_module(line)
            if not module:
                continue
            for part in _module_candidates(module):
                target = module_to_path.get(part)
                if target and target != path:
                    neighbors.append(target)
        graph[path] = _dedupe(neighbors)
    return graph


def dependency_closure(
    graph: dict[str, list[str]],
    roots: list[str],
    *,
    max_depth: int = 2,
    max_files: int = 24,
) -> list[str]:
    seen: list[str] = []
    seen_depth: dict[str, int] = {}
    frontier = [(root, 0) for root in roots]
    reverse = _reverse_graph(graph)
    while frontier and len(seen) < max_files:
        node, depth = frontier.pop(0)
        if node in seen_depth and seen_depth[node] <= depth:
            continue
        if node not in seen:
            seen.append(node)
        seen_depth[node] = depth
        if depth >= max_depth:
            continue
        for neighbor in _dedupe(graph.get(node, []) + reverse.get(node, [])):
            frontier.append((neighbor, depth + 1))
    return seen


def _reverse_graph(graph: dict[str, list[str]]) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for source, targets in graph.items():
        for target in targets:
            reverse.setdefault(target, []).append(source)
    return reverse


def _merge_graphs(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list[str]]:
    keys = set(a) | set(b)
    return {key: _dedupe(list(a.get(key, [])) + list(b.get(key, []))) for key in keys}


def _infer_service_boundaries(paths: Any) -> list[str]:
    boundaries = sorted({str(path).split("/", 1)[0] for path in paths if "/" in str(path)})
    return [boundary for boundary in boundaries if boundary not in {"tests", "__pycache__"}][:30]


def _module_name(path: str) -> str:
    return path.rsplit(".", 1)[0].replace("/", ".")


def _import_module(line: str) -> str | None:
    stripped = line.strip()
    from_match = re.match(r"from\s+([A-Za-z0-9_\.]+)\s+import\s+", stripped)
    import_match = re.match(r"import\s+([A-Za-z0-9_\.]+)", stripped)
    js_match = re.search(r"from\s+['\"]([^'\"]+)['\"]", stripped)
    match = from_match or import_match
    if match:
        return match.group(1)
    if js_match:
        return js_match.group(1).strip("./").replace("/", ".")
    return None


def _module_candidates(module: str) -> list[str]:
    parts = module.split(".")
    return [".".join(parts[:index]) for index in range(len(parts), 0, -1)]


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
