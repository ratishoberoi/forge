from __future__ import annotations
from dataclasses import dataclass, field
from backend.runtime.execution_node import ExecutionNode


@dataclass(slots=True)
class ExecutionGraph:
    nodes: dict[str, ExecutionNode] = field(default_factory=dict)

    def add_node(self, node: ExecutionNode) -> None:
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> ExecutionNode:
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in graph.")
        return self.nodes[node_id]

    def ready_nodes(self, completed: set[str]) -> list[ExecutionNode]:
        return [
            node for node in self.nodes.values()
            if node.node_id not in completed
            and all(dep in completed for dep in node.dependencies)
        ]

    @property
    def is_empty(self) -> bool:
        return not self.nodes

    @property
    def node_ids(self) -> list[str]:
        return list(self.nodes.keys())