from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from backend.runtime.json_store import atomic_write_json, load_json_store


@dataclass(slots=True)
class KnowledgeNode:
    node_id: str
    kind: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeNode:
        return cls(
            node_id=str(data["node_id"]),
            kind=str(data["kind"]),
            label=str(data["label"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class KnowledgeEdge:
    source: str
    target: str
    relationship: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeEdge:
        return cls(
            source=str(data["source"]),
            target=str(data["target"]),
            relationship=str(data["relationship"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class KnowledgeGraphRecord:
    repository_path: str
    nodes: list[KnowledgeNode] = field(default_factory=list)
    edges: list[KnowledgeEdge] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_path": self.repository_path,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "updated_at": self.updated_at,
            "stats": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "node_kinds": _counts(node.kind for node in self.nodes),
                "relationships": _counts(edge.relationship for edge in self.edges),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraphRecord:
        return cls(
            repository_path=str(data["repository_path"]),
            nodes=[KnowledgeNode.from_dict(item) for item in data.get("nodes", [])],
            edges=[KnowledgeEdge.from_dict(item) for item in data.get("edges", [])],
            updated_at=str(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
        )


class KnowledgeGraphStore:
    """Persistent local repository knowledge graph."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or ".forge/knowledge_graph.json").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def build_from_preparation(
        self,
        *,
        repository_path: str,
        objective: str,
        relevant_files: list[str],
        related_tests: list[str],
        dependency_graph: dict[str, list[str]],
        service_boundaries: list[str],
    ) -> KnowledgeGraphRecord:
        resolved_repo = str(Path(repository_path).resolve())
        existing = self.get(resolved_repo)
        nodes = {node.node_id: node for node in existing.nodes}
        edges = {
            (edge.source, edge.target, edge.relationship): edge
            for edge in existing.edges
        }
        feature_id = f"feature:{_slug(objective)}"
        nodes[feature_id] = KnowledgeNode(feature_id, "feature", objective, {"source": "objective"})
        for boundary in service_boundaries:
            service_id = f"service:{boundary}"
            nodes[service_id] = KnowledgeNode(service_id, "service", boundary)
        for file_path in relevant_files:
            file_id = f"file:{file_path}"
            nodes[file_id] = KnowledgeNode(file_id, "file", file_path, {"role": _file_role(file_path)})
            edges[(feature_id, file_id, "owns")] = KnowledgeEdge(feature_id, file_id, "owns")
            if "/" in file_path:
                service_id = f"service:{file_path.split('/', 1)[0]}"
                if service_id in nodes:
                    edges[(service_id, file_id, "owns")] = KnowledgeEdge(service_id, file_id, "owns")
        for test_path in related_tests:
            test_id = f"test:{test_path}"
            nodes[test_id] = KnowledgeNode(test_id, "test", test_path)
            for file_path in relevant_files[:20]:
                edges[(test_id, f"file:{file_path}", "tests")] = KnowledgeEdge(test_id, f"file:{file_path}", "tests")
        for source, targets in dependency_graph.items():
            source_id = f"file:{source}"
            if source_id not in nodes:
                continue
            for target in targets:
                target_id = f"file:{target}"
                nodes.setdefault(target_id, KnowledgeNode(target_id, "file", target, {"role": _file_role(target)}))
                edges[(source_id, target_id, "depends_on")] = KnowledgeEdge(source_id, target_id, "depends_on")
        record = KnowledgeGraphRecord(
            repository_path=resolved_repo,
            nodes=sorted(nodes.values(), key=lambda node: node.node_id),
            edges=sorted(edges.values(), key=lambda edge: (edge.source, edge.relationship, edge.target)),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        self._save_record(record)
        return record

    def get(self, repository_path: str) -> KnowledgeGraphRecord:
        key = str(Path(repository_path).resolve())
        return self._records().get(key) or KnowledgeGraphRecord(repository_path=key)

    def relevant_nodes(self, *, repository_path: str, query: str, limit: int = 20) -> list[KnowledgeNode]:
        record = self.get(repository_path)
        terms = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
        scored = []
        for node in record.nodes:
            node_terms = set(re.findall(r"[A-Za-z0-9_]+", f"{node.kind} {node.label}".lower()))
            score = len(terms & node_terms)
            if score:
                scored.append((score, node))
        return [node for _, node in sorted(scored, key=lambda item: (-item[0], item[1].node_id))[:limit]]

    def _save_record(self, record: KnowledgeGraphRecord) -> None:
        records = self._records()
        records[record.repository_path] = record
        payload = {"repositories": {key: value.to_dict() for key, value in records.items()}}
        atomic_write_json(self.path, payload)

    def _records(self) -> dict[str, KnowledgeGraphRecord]:
        data = load_json_store(
            self.path,
            default={"repositories": {}},
            store_name="knowledge_graph",
        )
        return {
            str(key): KnowledgeGraphRecord.from_dict(value)
            for key, value in dict(data.get("repositories", {})).items()
        }


def _file_role(path: str) -> str:
    lowered = path.lower()
    if lowered.startswith("tests/") or ".test." in lowered or ".spec." in lowered:
        return "test"
    if lowered.endswith((".md", ".rst")):
        return "documentation"
    if lowered.endswith((".json", ".toml", ".yaml", ".yml")):
        return "configuration"
    return "source"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "objective"


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result
