from __future__ import annotations
import uuid
from backend.runtime.objective import Objective
from backend.runtime.objective_graph import ObjectiveGraph
from backend.runtime.objective_state import ObjectiveState


class ObjectivePlanner:
    """
    Decomposes high-level engineering tasks into
    structured subgoal hierarchies.
    """

    def create_objective(
        self,
        *,
        title: str,
        description: str,
        subgoals: list[str],
    ) -> Objective:
        return Objective(
            objective_id=str(uuid.uuid4()),
            title=title,
            description=description,
            subgoals=subgoals,
        )

    def initialize_state(self, objective: Objective) -> ObjectiveState:
        """Create fresh ObjectiveState with all subgoals pending."""
        return ObjectiveState(pending=list(objective.subgoals))

    def next_subgoal(
        self,
        objective: Objective,
        state: ObjectiveState,
        graph: ObjectiveGraph,
    ) -> str | None:
        """Return next ready subgoal respecting dependency order."""
        ready = graph.ready_subgoals(
            all_subgoals=objective.subgoals,
            completed=state.completed,
        )
        pending_ready = [s for s in ready if s in state.pending]
        return pending_ready[0] if pending_ready else None