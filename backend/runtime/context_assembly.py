from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.runtime.adr import ADRRecord
from backend.runtime.knowledge_graph import KnowledgeGraphRecord, KnowledgeNode
from backend.runtime.project_brain import ProjectBrainRecord
from backend.runtime.repository_rag import RepositoryRAGHit
from backend.runtime.semantic_memory import SemanticMemoryItem


@dataclass(slots=True)
class AssembledContext:
    objective: str
    relevant_files: list[str] = field(default_factory=list)
    architecture_memory: dict[str, Any] | None = None
    project_brain: dict[str, Any] | None = None
    semantic_memories: list[dict[str, Any]] = field(default_factory=list)
    repository_hits: list[dict[str, Any]] = field(default_factory=list)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    previous_failures: list[str] = field(default_factory=list)
    previous_repairs: list[str] = field(default_factory=list)
    adrs: list[dict[str, Any]] = field(default_factory=list)
    knowledge_nodes: list[dict[str, Any]] = field(default_factory=list)
    context_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "relevant_files": self.relevant_files,
            "architecture_memory": self.architecture_memory,
            "project_brain": self.project_brain,
            "semantic_memories": self.semantic_memories,
            "repository_hits": self.repository_hits,
            "dependency_graph": self.dependency_graph,
            "previous_failures": self.previous_failures,
            "previous_repairs": self.previous_repairs,
            "adrs": self.adrs,
            "knowledge_nodes": self.knowledge_nodes,
            "context_usage": self.context_usage,
        }


class ContextAssemblyEngine:
    """Combines high-value local context for courtroom prompts."""

    def assemble(
        self,
        *,
        objective: str,
        relevant_files: list[str],
        dependency_graph: dict[str, list[str]],
        architecture_memory: Any,
        project_brain: ProjectBrainRecord,
        semantic_memories: list[SemanticMemoryItem],
        repository_hits: list[RepositoryRAGHit],
        adrs: list[ADRRecord],
        knowledge_graph: KnowledgeGraphRecord,
        knowledge_nodes: list[KnowledgeNode],
        max_files: int = 24,
    ) -> AssembledContext:
        file_scores: dict[str, float] = {path: 10.0 for path in relevant_files}
        for hit in repository_hits:
            if hit.path:
                file_scores[hit.path] = max(file_scores.get(hit.path, 0.0), hit.score * 10)
        for node in knowledge_nodes:
            if node.kind in {"file", "test"}:
                file_scores[node.label] = max(file_scores.get(node.label, 0.0), 7.0)
        selected = [
            path
            for path, _ in sorted(file_scores.items(), key=lambda item: (-item[1], item[0]))[:max_files]
        ]
        return AssembledContext(
            objective=objective,
            relevant_files=selected,
            architecture_memory=architecture_memory.to_dict() if architecture_memory else None,
            project_brain=project_brain.brief(objective=objective),
            semantic_memories=[item.to_dict() for item in semantic_memories],
            repository_hits=[hit.to_dict() for hit in repository_hits],
            dependency_graph={path: dependency_graph.get(path, []) for path in selected},
            previous_failures=project_brain.failures[-10:],
            previous_repairs=project_brain.repairs[-10:],
            adrs=[record.to_dict() for record in adrs[:12]],
            knowledge_nodes=[node.to_dict() for node in knowledge_nodes],
            context_usage={
                "selected_files": len(selected),
                "semantic_memories": len(semantic_memories),
                "repository_hits": len(repository_hits),
                "adrs": len(adrs),
                "knowledge_nodes": len(knowledge_nodes),
            },
        )
