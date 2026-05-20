from __future__ import annotations

import asyncio
from pathlib import Path

from backend.config.settings import Settings
from backend.runtime.adr import ADRStore
from backend.runtime.architecture_memory import ArchitectureMemory
from backend.runtime.context_assembly import ContextAssemblyEngine
from backend.runtime.knowledge_graph import KnowledgeGraphStore
from backend.runtime.project_brain import ProjectBrain
from backend.runtime.repository_execution_engine import RepositoryExecutionEngine
from backend.runtime.repository_execution_engine import ObjectiveType, classify_objective
from backend.runtime.repository_rag import RepositoryRAG
from backend.runtime.semantic_memory import SemanticMemory
from backend.runtime.tool_manager import ToolManager


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        repo_index_state_path=str(tmp_path / ".forge" / "index_state.json"),
        repo_incremental=False,
    )


def test_project_brain_persists_repository_knowledge(tmp_path: Path) -> None:
    brain = ProjectBrain(str(tmp_path / "project_brain.json"))
    repo = tmp_path / "repo"
    repo.mkdir()

    brain.update_from_preparation(
        repository_path=str(repo),
        objective="Add OAuth login",
        architecture_summary="FastAPI service with auth boundary.",
        service_boundaries=["app", "tests"],
        feature_summaries=["auth routes"],
    )
    brain.record_outcome(
        repository_path=str(repo),
        objective="Add OAuth login",
        implementation="app/auth.py",
        failures=["AssertionError"],
        repairs=["fixed callback"],
        successful=True,
    )

    reloaded = ProjectBrain(str(tmp_path / "project_brain.json")).get(str(repo))
    assert "Add OAuth login" in reloaded.previous_objectives
    assert "app/auth.py" in reloaded.successful_patterns
    assert "AssertionError" in reloaded.failures


def test_semantic_memory_retrieves_related_local_memories(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    memory = SemanticMemory(
        path=str(tmp_path / "semantic.sqlite3"),
        qdrant_path=str(tmp_path / "qdrant"),
    )

    memory.upsert(repository_path=str(repo), kind="implementation", text="JWT login callback stores auth session")
    memory.upsert(repository_path=str(repo), kind="failure", text="Stripe checkout route returned RuntimeError")

    hits = memory.retrieve(repository_path=str(repo), query="Add OAuth login and auth session", limit=2)

    assert hits
    assert hits[0].kind == "implementation"
    assert "login" in hits[0].text.lower()
    assert memory.stats(repository_path=str(repo))["items"] == 2


def test_repository_rag_indexes_docs_source_and_tests(tmp_path: Path) -> None:
    _write_auth_repo(tmp_path)
    memory = SemanticMemory(
        path=str(tmp_path / "semantic.sqlite3"),
        qdrant_path=str(tmp_path / "qdrant"),
    )
    rag = RepositoryRAG(repo_root=str(tmp_path), memory=memory)

    result = rag.index()
    hits = rag.retrieve(query="authentication login test", limit=4)

    assert result.indexed_files >= 4
    assert any(hit.path == "app/routes.py" for hit in hits)
    assert any(hit.metadata.get("role") == "test" for hit in hits)


def test_repository_rag_retrieval_is_isolated_by_repository_root(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "auth.py").write_text("def login():\n    return 'repo-a-auth'\n", encoding="utf-8")
    (repo_b / "billing.py").write_text("def checkout():\n    return 'repo-b-billing'\n", encoding="utf-8")
    memory = SemanticMemory(
        path=str(tmp_path / "semantic.sqlite3"),
        qdrant_path=str(tmp_path / "qdrant"),
    )
    rag_a = RepositoryRAG(repo_root=str(repo_a), memory=memory)
    rag_b = RepositoryRAG(repo_root=str(repo_b), memory=memory)

    rag_a.index()
    rag_b.index()
    hits_a = rag_a.retrieve(query="billing checkout auth", limit=10)
    hits_b = rag_b.retrieve(query="auth login billing", limit=10)

    assert hits_a
    assert hits_b
    assert all(hit.metadata.get("path") == "auth.py" for hit in hits_a)
    assert all(hit.metadata.get("path") == "billing.py" for hit in hits_b)


def test_corrupted_knowledge_graph_json_is_quarantined_and_rebuilt(tmp_path: Path) -> None:
    path = tmp_path / "knowledge_graph.json"
    path.write_text('{"repositories": {}}\n{"broken": true}', encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()

    graph = KnowledgeGraphStore(str(path)).build_from_preparation(
        repository_path=str(repo),
        objective="Add OAuth login",
        relevant_files=["app/routes.py"],
        related_tests=["tests/test_auth.py"],
        dependency_graph={},
        service_boundaries=["app"],
    )

    assert graph.repository_path == str(repo.resolve())
    assert path.exists()
    assert list(tmp_path.glob("knowledge_graph.json.corrupt-*.bak"))


def test_corrupted_semantic_memory_row_is_removed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    memory = SemanticMemory(path=str(tmp_path / "semantic.sqlite3"), qdrant_path=str(tmp_path / "qdrant"))
    item = memory.upsert(repository_path=str(repo), kind="objective", text="Add OAuth login")
    memory._conn.execute("UPDATE memories SET metadata = ? WHERE item_id = ?", ("{broken", item.item_id))
    memory._conn.commit()

    hits = memory.retrieve(repository_path=str(repo), query="OAuth", limit=5)

    assert hits == []
    assert memory.stats(repository_path=str(repo))["items"] == 0


def test_empty_fastapi_todo_objective_classifies_as_application() -> None:
    assert classify_objective("Build a complete FastAPI Todo application") == ObjectiveType.APPLICATION


def test_repository_preparation_builds_local_cognition_context(tmp_path: Path) -> None:
    _write_auth_repo(tmp_path)
    engine = RepositoryExecutionEngine(
        repo_root=str(tmp_path),
        settings=make_settings(tmp_path),
        architecture_memory=ArchitectureMemory(str(tmp_path / "architecture.json")),
    )

    prep = asyncio.run(engine.prepare("Add OAuth login"))

    assert prep.project_brain is not None
    assert "Add OAuth login" in prep.project_brain.previous_objectives
    assert prep.semantic_memories
    assert prep.repository_rag is not None
    assert prep.repository_rag["index"]["indexed_files"] >= 4
    assert prep.context_assembly is not None
    assert "app/routes.py" in prep.context_assembly.relevant_files
    assert prep.knowledge_graph is not None
    assert prep.knowledge_graph.to_dict()["stats"]["nodes"] > 0
    assert prep.adrs
    assert prep.tool_activity is not None
    assert "RepositoryTool" in prep.tool_activity["tools"]


def test_repository_preparation_recovers_corrupt_graph_and_uses_active_repo_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "app.py").write_text("def objective_summary():\n    return 'old bootstrap'\n", encoding="utf-8")
    (repo_b / "pyproject.toml").write_text("[project]\nname='todo'\n", encoding="utf-8")
    (tmp_path / ".forge").mkdir()
    (tmp_path / ".forge" / "knowledge_graph.json").write_text(
        '{"repositories": {}}\n{"partial": true}',
        encoding="utf-8",
    )
    memory = SemanticMemory(path=str(tmp_path / ".forge" / "semantic_memory.sqlite3"))
    memory.upsert(
        repository_path=str(repo_a),
        kind="repository_file",
        text="app/main.py\nobjective_summary()",
        metadata={"repository_path": str(repo_a.resolve()), "path": "app/main.py"},
    )
    engine = RepositoryExecutionEngine(repo_root=str(repo_b), settings=make_settings(tmp_path))

    prep = asyncio.run(engine.prepare("Build a complete FastAPI Todo application"))

    assert prep.plan.objective_type == ObjectiveType.APPLICATION
    assert "app/models.py" in prep.plan.files_to_create
    assert prep.knowledge_graph is not None
    assert prep.repository_rag is not None
    assert all("objective_summary" not in hit["content"] for hit in prep.repository_rag["hits"])
    assert list((tmp_path / ".forge").glob("knowledge_graph.json.corrupt-*.bak"))


def test_adr_knowledge_graph_and_context_assembly_are_local(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    adr_store = ADRStore(str(tmp_path / "adrs.json"))
    adrs = adr_store.infer_from_frameworks(
        repository_path=str(repo),
        frameworks=["FastAPI"],
        databases=["SQLite"],
    )
    graph = KnowledgeGraphStore(str(tmp_path / "graph.json")).build_from_preparation(
        repository_path=str(repo),
        objective="Add OAuth login",
        relevant_files=["app/routes.py"],
        related_tests=["tests/test_auth.py"],
        dependency_graph={"app/routes.py": ["app/services/auth.py"]},
        service_boundaries=["app", "tests"],
    )
    brain = ProjectBrain(str(tmp_path / "brain.json")).update_from_preparation(
        repository_path=str(repo),
        objective="Add OAuth login",
        architecture_summary="FastAPI app",
        service_boundaries=["app"],
    )
    memory = SemanticMemory(path=str(tmp_path / "semantic.sqlite3"), qdrant_path=str(tmp_path / "qdrant"))
    memory.upsert(repository_path=str(repo), kind="decision", text="Use FastAPI route conventions")

    assembled = ContextAssemblyEngine().assemble(
        objective="Add OAuth login",
        relevant_files=["app/routes.py"],
        dependency_graph={"app/routes.py": ["app/services/auth.py"]},
        architecture_memory=None,
        project_brain=brain,
        semantic_memories=memory.retrieve(repository_path=str(repo), query="OAuth FastAPI"),
        repository_hits=[],
        adrs=adrs,
        knowledge_graph=graph,
        knowledge_nodes=graph.nodes,
    )

    assert len(adrs) == 2
    assert graph.to_dict()["stats"]["relationships"]["depends_on"] == 1
    assert assembled.context_usage["semantic_memories"] == 1
    assert assembled.adrs


def test_tool_manager_observes_local_tool_activity(tmp_path: Path) -> None:
    _write_auth_repo(tmp_path)
    manager = ToolManager(repo_root=str(tmp_path))

    inspected = manager.inspect_repository()
    manager.search_repository("login", limit=2)

    assert inspected["files"] >= 4
    assert any(activity.tool == "RepositoryTool" for activity in manager.activities)
    assert any(activity.tool == "SearchTool" for activity in manager.activities)


def _write_auth_repo(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "[project]\nname='auth-app'\n[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Auth App\n\nLocal authentication documentation.\n", encoding="utf-8")
    (root / "app").mkdir()
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "routes.py").write_text(
        "from app.services.auth import authenticate\n\n"
        "def login(email: str, password: str):\n"
        "    return authenticate(email, password)\n",
        encoding="utf-8",
    )
    (root / "app" / "services").mkdir()
    (root / "app" / "services" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "services" / "auth.py").write_text(
        "def authenticate(email: str, password: str):\n"
        "    return {'email': email, 'token': 'local'}\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_auth.py").write_text(
        "from app.routes import login\n\n"
        "def test_login():\n"
        "    assert login('a@example.com', 'pw')['token'] == 'local'\n",
        encoding="utf-8",
    )
