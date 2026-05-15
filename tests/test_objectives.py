from backend.runtime.objective_graph import ObjectiveGraph
from backend.runtime.objective_planner import ObjectivePlanner
from backend.runtime.objective_state import ObjectiveState


def test_create_objective():
    planner = ObjectivePlanner()
    objective = planner.create_objective(
        title="Fix auth",
        description="Repair auth flow",
        subgoals=["inspect routes", "validate middleware"],
    )
    assert objective.title == "Fix auth"
    assert len(objective.subgoals) == 2
    assert objective.subgoal_count == 2
    assert not objective.is_empty


def test_objective_id_is_unique():
    planner = ObjectivePlanner()
    a = planner.create_objective(title="A", description="A", subgoals=[])
    b = planner.create_objective(title="B", description="B", subgoals=[])
    assert a.objective_id != b.objective_id


def test_objective_state():
    state = ObjectiveState()
    state.pending.append("inspect routes")
    state.completed.append("validate middleware")
    assert len(state.pending) == 1
    assert len(state.completed) == 1


def test_objective_state_mark_complete():
    state = ObjectiveState(pending=["inspect routes", "patch bug"])
    state.mark_complete("inspect routes")
    assert "inspect routes" in state.completed
    assert "inspect routes" not in state.pending


def test_objective_state_mark_failed():
    state = ObjectiveState(pending=["inspect routes"])
    state.mark_failed("inspect routes")
    assert "inspect routes" in state.failed
    assert "inspect routes" not in state.pending


def test_objective_state_progress_ratio():
    state = ObjectiveState(
        completed=["a", "b"],
        pending=["c"],
        failed=[],
    )
    assert state.progress_ratio == 2 / 3


def test_objective_state_is_complete():
    state = ObjectiveState(completed=["a"], pending=[], failed=[])
    assert state.is_complete


def test_objective_state_has_failures():
    state = ObjectiveState(failed=["a"])
    assert state.has_failures


def test_objective_graph_dependencies():
    graph = ObjectiveGraph()
    graph.add_dependency(subgoal="run tests", depends_on="patch bug")
    assert "patch bug" in graph.dependencies_for("run tests")


def test_objective_graph_ready_subgoals():
    graph = ObjectiveGraph()
    graph.add_dependency(subgoal="run tests", depends_on="patch bug")
    ready = graph.ready_subgoals(
        all_subgoals=["patch bug", "run tests"],
        completed=["patch bug"],
    )
    assert "run tests" in ready
    assert "patch bug" not in ready


def test_objective_graph_no_deps_ready_immediately():
    graph = ObjectiveGraph()
    ready = graph.ready_subgoals(
        all_subgoals=["inspect routes"],
        completed=[],
    )
    assert "inspect routes" in ready


def test_planner_initialize_state():
    planner = ObjectivePlanner()
    objective = planner.create_objective(
        title="Fix auth",
        description="desc",
        subgoals=["a", "b", "c"],
    )
    state = planner.initialize_state(objective)
    assert state.pending == ["a", "b", "c"]
    assert state.completed == []


def test_planner_next_subgoal_respects_deps():
    planner = ObjectivePlanner()
    graph = ObjectiveGraph()
    graph.add_dependency(subgoal="run tests", depends_on="patch bug")
    objective = planner.create_objective(
        title="Fix",
        description="desc",
        subgoals=["patch bug", "run tests"],
    )
    state = planner.initialize_state(objective)
    next_sg = planner.next_subgoal(objective, state, graph)
    assert next_sg == "patch bug"