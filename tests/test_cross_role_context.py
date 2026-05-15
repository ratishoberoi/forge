import pytest
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_loader import ArtifactLoader
from backend.runtime.artifact_merge import ArtifactMerger
from backend.runtime.artifact_query import ArtifactQueryEngine
from backend.runtime.artifact_store import ArtifactStore
from backend.runtime.cross_role_context import CrossRoleContextBuilder


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_artifact(
    artifact_id: str,
    role: str,
    round_id: int,
    content: str,
    task: str = "Task",
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


def make_builder(tmp_path) -> CrossRoleContextBuilder:
    return CrossRoleContextBuilder(
        query_engine=ArtifactQueryEngine(
            ArtifactLoader(base_dir=str(tmp_path))
        ),
        merger=ArtifactMerger(),
    )


def seed_store(tmp_path, artifacts: list[CognitionArtifact]) -> None:
    store = ArtifactStore(base_dir=str(tmp_path))
    for artifact in artifacts:
        store.save(artifact)


# ── build ─────────────────────────────────────────────────────────────────────

def test_cross_role_context_building(tmp_path):
    seed_store(tmp_path, [
        make_artifact("a1", "PRIMARY_CODER", 1, "Primary implementation"),
        make_artifact("a2", "JUDGE", 1, "Judge critique"),
    ])

    builder = make_builder(tmp_path)
    context = builder.build(["PRIMARY_CODER", "JUDGE"])

    assert "Primary implementation" in context
    assert "Judge critique" in context


def test_deterministic_multi_role_ordering(tmp_path):
    seed_store(tmp_path, [
        make_artifact("a2", "JUDGE", 2, "Judge round 2"),
        make_artifact("a1", "PRIMARY_CODER", 1, "Primary round 1"),
    ])

    builder = make_builder(tmp_path)
    context = builder.build(["PRIMARY_CODER", "JUDGE"])

    assert context.index("Primary round 1") < context.index("Judge round 2")


def test_build_empty_roles_returns_empty(tmp_path):
    builder = make_builder(tmp_path)
    context = builder.build([])
    assert context == ""


def test_build_missing_role_returns_empty(tmp_path):
    builder = make_builder(tmp_path)
    context = builder.build(["NON_EXISTENT"])
    assert context == ""


# ── build_for_round ───────────────────────────────────────────────────────────

def test_build_for_round(tmp_path):
    seed_store(tmp_path, [
        make_artifact("a1", "PRIMARY_CODER", 1, "Round 1 content"),
        make_artifact("a2", "PRIMARY_CODER", 2, "Round 2 content"),
    ])

    builder = make_builder(tmp_path)
    context = builder.build_for_round(["PRIMARY_CODER"], 1)

    assert "Round 1 content" in context
    assert "Round 2 content" not in context


# ── build_for_round_range ─────────────────────────────────────────────────────

def test_build_for_round_range(tmp_path):
    seed_store(tmp_path, [
        make_artifact("a1", "PRIMARY_CODER", 1, "Round 1"),
        make_artifact("a2", "PRIMARY_CODER", 2, "Round 2"),
        make_artifact("a3", "PRIMARY_CODER", 3, "Round 3"),
    ])

    builder = make_builder(tmp_path)
    context = builder.build_for_round_range(["PRIMARY_CODER"], start=1, end=2)

    assert "Round 1" in context
    assert "Round 2" in context
    assert "Round 3" not in context


# ── build_latest ──────────────────────────────────────────────────────────────

def test_build_latest(tmp_path):
    seed_store(tmp_path, [
        make_artifact(f"a{i}", "PRIMARY_CODER", i, f"Round {i}")
        for i in range(1, 6)
    ])

    builder = make_builder(tmp_path)
    context = builder.build_latest(["PRIMARY_CODER"], n_rounds=2)

    assert "Round 4" in context
    assert "Round 5" in context
    assert "Round 1" not in context


# ── build_by_metadata ─────────────────────────────────────────────────────────

def test_build_by_metadata(tmp_path):
    seed_store(tmp_path, [
        make_artifact("a1", "PRIMARY_CODER", 1, "Agent 1 content", metadata={"agent_id": "coder-1"}),
        make_artifact("a2", "PRIMARY_CODER", 2, "Agent 2 content", metadata={"agent_id": "coder-2"}),
    ])

    builder = make_builder(tmp_path)
    context = builder.build_by_metadata(["PRIMARY_CODER"], agent_id="coder-1")

    assert "Agent 1 content" in context
    assert "Agent 2 content" not in context


# ── build_by_content ──────────────────────────────────────────────────────────

def test_build_by_content(tmp_path):
    seed_store(tmp_path, [
        make_artifact("a1", "PRIMARY_CODER", 1, "def hello(name: str) -> str"),
        make_artifact("a2", "PRIMARY_CODER", 2, "def goodbye(name: str) -> str"),
    ])

    builder = make_builder(tmp_path)
    context = builder.build_by_content(["PRIMARY_CODER"], "hello")

    assert "hello" in context
    assert "goodbye" not in context


# ── latest_per_role ───────────────────────────────────────────────────────────

def test_latest_per_role(tmp_path):
    seed_store(tmp_path, [
        make_artifact("a1", "PRIMARY_CODER", 1, "Round 1"),
        make_artifact("a2", "PRIMARY_CODER", 2, "Round 2"),
        make_artifact("a3", "JUDGE", 1, "Judge round 1"),
    ])

    builder = make_builder(tmp_path)
    latest = builder.latest_per_role(["PRIMARY_CODER", "JUDGE"])

    assert latest["PRIMARY_CODER"].round_id == 2
    assert latest["JUDGE"].round_id == 1


# ── roles_present ─────────────────────────────────────────────────────────────

def test_roles_present(tmp_path):
    seed_store(tmp_path, [
        make_artifact("a1", "PRIMARY_CODER", 1, "content"),
        make_artifact("a2", "JUDGE", 1, "content"),
    ])

    builder = make_builder(tmp_path)
    roles = builder.roles_present(["PRIMARY_CODER", "JUDGE", "NON_EXISTENT"])

    assert "PRIMARY_CODER" in roles
    assert "JUDGE" in roles
    assert "NON_EXISTENT" not in roles