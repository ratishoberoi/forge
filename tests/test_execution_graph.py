from backend.runtime.execution_graph import ExecutionGraph
from backend.runtime.execution_node import ExecutionNode
from backend.runtime.graph_scheduler import GraphScheduler
from backend.runtime.graph_state import GraphExecutionState


def build_node(node_id: str, deps: list[str] | None = None) -> ExecutionNode:
    return ExecutionNode(node_id=node_id, title=node_id.title(), dependencies=deps or [])


def test_ready_nodes_without_dependencies():
    graph = ExecutionGraph()
    graph.add_node(build_node("a"))
    nodes = GraphScheduler().next_nodes(graph=graph, state=GraphExecutionState())
    assert len(nodes) == 1
    assert nodes[0].node_id == "a"


def test_dependency_blocks_execution():
    graph = ExecutionGraph()
    graph.add_node(build_node("a"))
    graph.add_node(build_node("b", deps=["a"]))
    ids = [n.node_id for n in GraphScheduler().next_nodes(graph=graph, state=GraphExecutionState())]
    assert "a" in ids
    assert "b" not in ids


def test_dependency_unlocks_after_completion():
    graph = ExecutionGraph()
    graph.add_node(build_node("a"))
    graph.add_node(build_node("b", deps=["a"]))
    state = GraphExecutionState()
    state.completed.add("a")
    ids = [n.node_id for n in GraphScheduler().next_nodes(graph=graph, state=state)]
    assert "b" in ids
    assert "a" not in ids


def test_multiple_ready_nodes():
    graph = ExecutionGraph()
    graph.add_node(build_node("a"))
    graph.add_node(build_node("b"))
    graph.add_node(build_node("c", deps=["a"]))
    ids = [n.node_id for n in GraphScheduler().next_nodes(graph=graph, state=GraphExecutionState())]
    assert "a" in ids
    assert "b" in ids
    assert "c" not in ids


def test_is_complete():
    graph = ExecutionGraph()
    graph.add_node(build_node("a"))
    graph.add_node(build_node("b"))
    scheduler = GraphScheduler()
    state = GraphExecutionState()
    assert not scheduler.is_complete(graph=graph, state=state)
    state.mark_complete("a")
    state.mark_complete("b")
    assert scheduler.is_complete(graph=graph, state=state)


def test_is_deadlocked():
    graph = ExecutionGraph()
    graph.add_node(build_node("a"))
    graph.add_node(build_node("b", deps=["a"]))
    scheduler = GraphScheduler()
    state = GraphExecutionState()
    state.failed.add("a")
    assert scheduler.is_deadlocked(graph=graph, state=state)


def test_mark_complete_removes_from_failed():
    state = GraphExecutionState()
    state.mark_failed("a")
    assert "a" in state.failed
    state.mark_complete("a")
    assert "a" in state.completed
    assert "a" not in state.failed


def test_empty_graph_is_complete():
    graph = ExecutionGraph()
    scheduler = GraphScheduler()
    assert scheduler.is_complete(graph=graph, state=GraphExecutionState())


def test_get_node_missing_raises():
    graph = ExecutionGraph()
    try:
        graph.get_node("ghost")
        assert False, "Should have raised"
    except KeyError:
        pass