from backend.runtime.repo_analysis import RepositoryImpactAnalyzer
from backend.runtime.repo_boundary import RepositoryBoundaryPolicy
from backend.runtime.repo_file import RepositoryFile
from backend.runtime.repo_graph import RepositoryGraph


def build_graph() -> RepositoryGraph:
    graph = RepositoryGraph()
    graph.add_dependency(source="auth", target="database")
    graph.add_dependency(source="database", target="cache")
    return graph


def test_repository_graph_neighbors():
    graph = build_graph()
    assert "database" in graph.neighbors("auth")


def test_repository_graph_unknown_module_empty():
    graph = RepositoryGraph()
    assert graph.neighbors("ghost") == set()


def test_repository_graph_all_modules():
    graph = build_graph()
    modules = graph.all_modules()
    assert "auth" in modules
    assert "database" in modules
    assert "cache" in modules


def test_repository_graph_reverse():
    graph = build_graph()
    rev = graph.reverse()
    assert "auth" in rev.neighbors("database")
    assert "database" in rev.neighbors("cache")


def test_impact_analysis():
    analyzer = RepositoryImpactAnalyzer()
    impacted = analyzer.impacted_modules(graph=build_graph(), module="auth")
    assert "auth" in impacted
    assert "database" in impacted
    assert "cache" in impacted


def test_impact_analysis_isolated_module():
    graph = RepositoryGraph()
    graph.add_dependency(source="a", target="b")
    analyzer = RepositoryImpactAnalyzer()
    impacted = analyzer.impacted_modules(graph=graph, module="b")
    assert impacted == {"b"}


def test_blast_radius():
    analyzer = RepositoryImpactAnalyzer()
    assert analyzer.blast_radius(graph=build_graph(), module="auth") == 3
    assert analyzer.blast_radius(graph=build_graph(), module="cache") == 1


def test_reverse_impact():
    analyzer = RepositoryImpactAnalyzer()
    dependents = analyzer.reverse_impact(graph=build_graph(), module="database")
    assert "auth" in dependents


def test_boundary_policy_allows():
    policy = RepositoryBoundaryPolicy(protected_modules={"core_runtime"})
    assert not policy.allows("core_runtime")
    assert policy.allows("feature_module")


def test_boundary_policy_protect():
    policy = RepositoryBoundaryPolicy()
    policy.protect("core_runtime", "auth_core")
    assert not policy.allows("core_runtime")
    assert not policy.allows("auth_core")


def test_boundary_violation_reason():
    policy = RepositoryBoundaryPolicy(protected_modules={"core_runtime"})
    assert policy.violation_reason("core_runtime") is not None
    assert policy.violation_reason("safe_module") is None


def test_repo_file_properties():
    f = RepositoryFile(path="backend/auth/service.py", module="auth")
    assert f.is_python
    assert f.extension == "py"


def test_repo_file_no_extension():
    f = RepositoryFile(path="Makefile", module="build")
    assert f.extension == ""
    assert not f.is_python