import pytest
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_context import ArtifactContextBuilder
from backend.runtime.artifact_loader import ArtifactLoader
from backend.runtime.artifact_store import ArtifactStore


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_artifact(
    artifact_id: str,
    role: str,
    round_id: int,
    content: str,
    task: str = "Test task",
    metadata: dict | None = None,
) -> CognitionArtifact:
    return CognitionArtifact(
        artifact_id=artifact_id,
        role=role,
        round_id=round_id,
        task=task,
        content=content,
        metadata=metadata or {},
    )


# ── ArtifactStore ─────────────────────────────────────────────────────────────

def test_save_and_load_artifact(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    artifact = make_artifact("a1", "PRIMARY_CODER", 1, "Use str type hints.")

    store.save(artifact)

    loader = ArtifactLoader(base_dir=str(tmp_path))
    loaded = loader.load_role_artifacts("PRIMARY_CODER")

    assert len(loaded) == 1
    assert loaded[0].artifact_id == "a1"
    assert loaded[0].content == "Use str type hints."


def test_save_returns_path(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    artifact = make_artifact("a1", "PRIMARY_CODER", 1, "content")

    path = store.save(artifact)

    assert path.exists()
    assert path.suffix == ".json"


def test_store_exists(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    artifact = make_artifact("a1", "PRIMARY_CODER", 1, "content")
    store.save(artifact)

    assert store.exists("PRIMARY_CODER", 1) is True
    assert store.exists("PRIMARY_CODER", 99) is False


def test_store_load_single(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    store.save(make_artifact("a1", "PRIMARY_CODER", 1, "content"))

    loaded = store.load("PRIMARY_CODER", 1)
    assert loaded.artifact_id == "a1"


def test_store_load_missing_raises(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    with pytest.raises(Exception, match="not found"):
        store.load("PRIMARY_CODER", 99)


def test_store_delete(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    store.save(make_artifact("a1", "PRIMARY_CODER", 1, "content"))

    deleted = store.delete("PRIMARY_CODER", 1)
    assert deleted is True
    assert store.exists("PRIMARY_CODER", 1) is False


def test_store_delete_missing_returns_false(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    assert store.delete("PRIMARY_CODER", 99) is False


def test_store_clear_role(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    for i in range(3):
        store.save(make_artifact(f"a{i}", "PRIMARY_CODER", i, f"content {i}"))

    count = store.clear_role("PRIMARY_CODER")
    assert count == 3
    assert store.load_all("PRIMARY_CODER") == []


# ── ArtifactLoader ────────────────────────────────────────────────────────────

def test_role_separation(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    store.save(make_artifact("a1", "PRIMARY_CODER", 1, "Primary content"))
    store.save(make_artifact("a2", "JUDGE", 1, "Judge content"))

    loader = ArtifactLoader(base_dir=str(tmp_path))
    primary = loader.load_role_artifacts("PRIMARY_CODER")
    judge = loader.load_role_artifacts("JUDGE")

    assert len(primary) == 1
    assert len(judge) == 1
    assert primary[0].role == "PRIMARY_CODER"
    assert judge[0].role == "JUDGE"


def test_deterministic_ordering(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    for round_id in [3, 1, 2]:
        store.save(make_artifact(f"a{round_id}", "PRIMARY_CODER", round_id, f"Round {round_id}"))

    loader = ArtifactLoader(base_dir=str(tmp_path))
    loaded = loader.load_role_artifacts("PRIMARY_CODER")
    rounds = [a.round_id for a in loaded]

    assert rounds == [1, 2, 3]


def test_missing_role_returns_empty(tmp_path):
    loader = ArtifactLoader(base_dir=str(tmp_path))
    loaded = loader.load_role_artifacts("NON_EXISTENT")
    assert loaded == []


def test_loader_load_round(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    store.save(make_artifact("a2", "PRIMARY_CODER", 2, "Round 2 content"))

    loader = ArtifactLoader(base_dir=str(tmp_path))
    artifact = loader.load_round("PRIMARY_CODER", 2)

    assert artifact.artifact_id == "a2"
    assert artifact.content == "Round 2 content"


def test_loader_load_round_missing_raises(tmp_path):
    loader = ArtifactLoader(base_dir=str(tmp_path))
    with pytest.raises(Exception, match="not found"):
        loader.load_round("PRIMARY_CODER", 99)


def test_loader_load_round_range(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    for i in range(1, 6):
        store.save(make_artifact(f"a{i}", "PRIMARY_CODER", i, f"Round {i}"))

    loader = ArtifactLoader(base_dir=str(tmp_path))
    loaded = loader.load_round_range("PRIMARY_CODER", start=2, end=4)
    rounds = [a.round_id for a in loaded]

    assert rounds == [2, 3, 4]


def test_loader_load_by_metadata(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    store.save(make_artifact("a1", "PRIMARY_CODER", 1, "content", metadata={"agent_id": "coder-1"}))
    store.save(make_artifact("a2", "PRIMARY_CODER", 2, "content", metadata={"agent_id": "coder-2"}))

    loader = ArtifactLoader(base_dir=str(tmp_path))
    loaded = loader.load_by_metadata("PRIMARY_CODER", agent_id="coder-1")

    assert len(loaded) == 1
    assert loaded[0].artifact_id == "a1"


def test_loader_list_roles(tmp_path):
    store = ArtifactStore(base_dir=str(tmp_path))
    store.save(make_artifact("a1", "PRIMARY_CODER", 1, "content"))
    store.save(make_artifact("a2", "JUDGE", 1, "content"))

    loader = ArtifactLoader(base_dir=str(tmp_path))
    roles = loader.list_roles()

    assert "primary_coder" in roles
    assert "judge" in roles


# ── ArtifactContextBuilder ────────────────────────────────────────────────────

def test_context_builder_basic(tmp_path):
    builder = ArtifactContextBuilder()
    context = builder.build_context([
        make_artifact("a1", "PRIMARY_CODER", 1, "Initial implementation"),
        make_artifact("a2", "JUDGE", 1, "Critique implementation"),
    ])

    assert "[ROLE: PRIMARY_CODER]" in context
    assert "[ROLE: JUDGE]" in context
    assert "Initial implementation" in context
    assert "Critique implementation" in context


def test_context_builder_empty_returns_empty_string():
    builder = ArtifactContextBuilder()
    assert builder.build_context([]) == ""


def test_context_builder_truncation():
    builder = ArtifactContextBuilder()
    artifact = make_artifact("a1", "PRIMARY_CODER", 1, "x" * 50_000)
    context = builder.build_context([artifact], max_chars=100)

    assert len(context) <= 100 + len("\n\n[... context truncated ...]")
    assert "truncated" in context


def test_context_builder_role_filter():
    builder = ArtifactContextBuilder()
    artifacts = [
        make_artifact("a1", "PRIMARY_CODER", 1, "Coder content"),
        make_artifact("a2", "JUDGE", 2, "Judge content"),
    ]
    context = builder.build_role_context(artifacts, "JUDGE")

    assert "Judge content" in context
    assert "Coder content" not in context


def test_context_builder_latest_n():
    builder = ArtifactContextBuilder()
    artifacts = [make_artifact(f"a{i}", "PRIMARY_CODER", i, f"Round {i}") for i in range(1, 6)]
    context = builder.build_latest_context(artifacts, n=2)

    assert "Round 4" in context
    assert "Round 5" in context
    assert "Round 1" not in context


def test_context_builder_summarize():
    builder = ArtifactContextBuilder()
    artifacts = [make_artifact("a1", "PRIMARY_CODER", 1, "content")]
    summary = builder.summarize(artifacts)

    assert "round=1" in summary
    assert "PRIMARY_CODER" in summary