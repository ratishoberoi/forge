import pytest
from datetime import datetime, timezone
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_exchange import ArtifactExchange, ArtifactExchangeError
from backend.runtime.artifact_loader import ArtifactLoader
from backend.runtime.artifact_revision import ArtifactRevision
from backend.runtime.artifact_store import ArtifactStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_exchange(tmp_path) -> ArtifactExchange:
    return ArtifactExchange(
        store=ArtifactStore(str(tmp_path)),
        loader=ArtifactLoader(str(tmp_path)),
    )


def build_artifact(
    artifact_id: str = "a1",
    role: str = "coder",
    round_id: int = 1,
    content: str = "patch_v1",
    task: str = "auth patch",
) -> CognitionArtifact:
    return CognitionArtifact(
        artifact_id=artifact_id,
        role=role,
        round_id=round_id,
        task=task,
        content=content,
        created_at=datetime.now(timezone.utc),
        metadata={},
    )


# ── ArtifactRevision ──────────────────────────────────────────────────────────

def test_revision_is_root():
    rev = ArtifactRevision.create(
        artifact_id="a1",
        role="coder",
        summary="Initial implementation",
    )
    assert rev.is_root is True
    assert rev.parent_revision_id is None


def test_revision_child_linkage():
    root = ArtifactRevision.create(
        artifact_id="a1",
        role="coder",
        summary="Initial implementation",
    )
    child = ArtifactRevision.create(
        artifact_id="a1",
        role="judge",
        summary="Revised after critique",
        parent_revision_id=root.revision_id,
    )
    assert child.is_root is False
    assert child.parent_revision_id == root.revision_id


def test_revision_blank_summary_raises():
    with pytest.raises(ValueError, match="blank"):
        ArtifactRevision(
            revision_id="r1",
            artifact_id="a1",
            role="coder",
            summary="   ",
            created_at=datetime.now(timezone.utc),
        )


def test_revision_empty_role_raises():
    with pytest.raises(ValueError, match="role"):
        ArtifactRevision(
            revision_id="r1",
            artifact_id="a1",
            role="",
            summary="summary",
            created_at=datetime.now(timezone.utc),
        )


def test_revision_factory_autogenerates_id():
    rev = ArtifactRevision.create(
        artifact_id="a1",
        role="coder",
        summary="Auto revision",
    )
    assert rev.revision_id != ""
    assert len(rev.revision_id) == 36  # uuid4


def test_revision_to_dict():
    rev = ArtifactRevision.create(
        artifact_id="a1",
        role="judge",
        summary="Critique applied",
        parent_revision_id="parent-uuid",
    )
    d = rev.to_dict()
    assert d["artifact_id"] == "a1"
    assert d["role"] == "judge"
    assert d["parent_revision_id"] == "parent-uuid"
    assert "created_at" in d


# ── ArtifactExchange core ─────────────────────────────────────────────────────

def test_exchange_roundtrip(tmp_path):
    exchange = make_exchange(tmp_path)
    artifact = build_artifact()

    exchange.persist(artifact)
    loaded = exchange.retrieve_round(role="coder", round_id=1)

    assert loaded.content == "patch_v1"
    assert loaded.artifact_id == "a1"


def test_exchange_persist_returns_path(tmp_path):
    exchange = make_exchange(tmp_path)
    path = exchange.persist(build_artifact())

    assert path.exists()
    assert path.suffix == ".json"


def test_exchange_blank_content_raises(tmp_path):
    exchange = make_exchange(tmp_path)
    artifact = build_artifact(content="   ")

    with pytest.raises(ArtifactExchangeError, match="blank content"):
        exchange.persist(artifact)


def test_exchange_retrieve_missing_round_raises(tmp_path):
    exchange = make_exchange(tmp_path)

    with pytest.raises(ArtifactExchangeError):
        exchange.retrieve_round(role="coder", round_id=99)


# ── Role history ──────────────────────────────────────────────────────────────

def test_role_history_single(tmp_path):
    exchange = make_exchange(tmp_path)
    exchange.persist(build_artifact())

    history = exchange.retrieve_role_history("coder")

    assert len(history) == 1
    assert history[0].content == "patch_v1"


def test_role_history_multiple_rounds(tmp_path):
    exchange = make_exchange(tmp_path)
    exchange.persist(build_artifact(round_id=1, content="v1"))
    exchange.persist(build_artifact(artifact_id="a2", round_id=2, content="v2"))
    exchange.persist(build_artifact(artifact_id="a3", round_id=3, content="v3"))

    history = exchange.retrieve_role_history("coder")

    assert len(history) == 3
    assert [a.round_id for a in history] == [1, 2, 3]


def test_role_history_empty_returns_empty(tmp_path):
    exchange = make_exchange(tmp_path)
    history = exchange.retrieve_role_history("nonexistent")
    assert history == []


def test_role_history_sorted_by_round(tmp_path):
    exchange = make_exchange(tmp_path)
    for round_id in [3, 1, 2]:
        exchange.persist(
            build_artifact(artifact_id=f"a{round_id}", round_id=round_id)
        )

    history = exchange.retrieve_role_history("coder")
    assert [a.round_id for a in history] == [1, 2, 3]


# ── retrieve_latest ───────────────────────────────────────────────────────────

def test_retrieve_latest(tmp_path):
    exchange = make_exchange(tmp_path)
    exchange.persist(build_artifact(artifact_id="a1", round_id=1, content="v1"))
    exchange.persist(build_artifact(artifact_id="a2", round_id=2, content="v2"))

    latest = exchange.retrieve_latest("coder")
    assert latest.content == "v2"
    assert latest.round_id == 2


def test_retrieve_latest_none_when_empty(tmp_path):
    exchange = make_exchange(tmp_path)
    assert exchange.retrieve_latest("coder") is None


# ── exists ────────────────────────────────────────────────────────────────────

def test_exists_true(tmp_path):
    exchange = make_exchange(tmp_path)
    exchange.persist(build_artifact())
    assert exchange.exists("coder", 1) is True


def test_exists_false(tmp_path):
    exchange = make_exchange(tmp_path)
    assert exchange.exists("coder", 99) is False


# ── Audit log ─────────────────────────────────────────────────────────────────

def test_persist_count(tmp_path):
    exchange = make_exchange(tmp_path)
    exchange.persist(build_artifact(artifact_id="a1", round_id=1))
    exchange.persist(build_artifact(artifact_id="a2", round_id=2))
    assert exchange.persist_count == 2


def test_persisted_ids(tmp_path):
    exchange = make_exchange(tmp_path)
    exchange.persist(build_artifact(artifact_id="a1", round_id=1))
    exchange.persist(build_artifact(artifact_id="a2", round_id=2))
    assert exchange.persisted_ids == ["a1", "a2"]


# ── Runtime-to-runtime simulation ─────────────────────────────────────────────

def test_coder_to_judge_handoff(tmp_path):
    """Coder runtime persists → judge runtime retrieves."""
    coder_exchange = make_exchange(tmp_path)
    judge_exchange = make_exchange(tmp_path)

    coder_exchange.persist(build_artifact(
        content="def hello(name: str) -> str: return 'hello ' + name"
    ))

    loaded = judge_exchange.retrieve_round(role="coder", round_id=1)
    assert "name: str" in loaded.content


def test_multi_hop_cognition_chain(tmp_path):
    """Coder → Judge → Synth chain via artifact exchange."""
    exchange = make_exchange(tmp_path)

    # Round 1: Coder writes
    exchange.persist(build_artifact(
        artifact_id="coder_r1", role="coder", round_id=1,
        content="Initial: def hello(name): return name",
    ))

    # Round 2: Judge reads coder, writes critique
    coder_out = exchange.retrieve_round(role="coder", round_id=1)
    exchange.persist(CognitionArtifact(
        artifact_id="judge_r1", role="judge", round_id=1,
        task="critique", content=f"Missing type hints in: {coder_out.content[:20]}",
        created_at=datetime.now(timezone.utc), metadata={},
    ))

    # Round 3: Synth reads judge, writes synthesis
    judge_out = exchange.retrieve_round(role="judge", round_id=1)
    exchange.persist(CognitionArtifact(
        artifact_id="synth_r1", role="synth", round_id=1,
        task="synthesize", content=f"Synthesis: {judge_out.content[:20]} — add str hints.",
        created_at=datetime.now(timezone.utc), metadata={},
    ))

    synth_out = exchange.retrieve_round(role="synth", round_id=1)
    assert "Synthesis" in synth_out.content
    assert exchange.persist_count == 3