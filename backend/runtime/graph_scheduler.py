from __future__ import annotations
from backend.runtime.execution_graph import ExecutionGraph
from backend.runtime.execution_node import ExecutionNode
from backend.runtime.graph_state import GraphExecutionState


class GraphScheduler:
    """
    Dependency-aware execution scheduler.
    Responsibilities:
    - surface ready nodes respecting dependency order
    - detect execution completion
    - detect deadlocks (pending nodes with unresolvable deps)
    """

    def next_nodes(
        self,
        *,
        graph: ExecutionGraph,
        state: GraphExecutionState,
    ) -> list[ExecutionNode]:
        excluded = state.completed | state.failed
        return [
            node for node in graph.nodes.values()
            if node.node_id not in excluded
            and all(dep in state.completed for dep in node.dependencies)
        ]

    def is_complete(
        self,
        *,
        graph: ExecutionGraph,
        state: GraphExecutionState,
    ) -> bool:
        """True when all nodes are completed."""
        return set(graph.node_ids) == state.completed

    def is_deadlocked(
        self,
        *,
        graph: ExecutionGraph,
        state: GraphExecutionState,
    ) -> bool:
        pending = set(graph.node_ids) - state.completed - state.failed
        if not pending:
            return False
        return len(self.next_nodes(graph=graph, state=state)) == 0