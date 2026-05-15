import pytest
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_replay import ArtifactReplayEngine
from backend.runtime.artifact_timeline import ArtifactTimeline
from backend.runtime.replay_context import ReplayContextBuilder
from backend.runtime.timeline_policy import TimelinePolicy


def make_artifact(role: str, round_id: int, content: str = "") -> CognitionArtifact:
    return CognitionArtifact(
        artifact_id=f"{role}-{round_id}",
        role=role,
        round_id=round_id,
        task="Task",
        content=content or f"Artifact {round_id}",
    )


def build_artifacts(count: int, role: str = "PRIMARY_CODER") -> list[CognitionArtifact]:
    return [make_artifact(role, i) for i in range(count)]


# ── ArtifactTimeline ──────────────────────────────────────────────────────────

def test_timeline_ordering():
    timeline = ArtifactTimeline([
        make_artifact("PRIMARY_CODER", 2, "Round 2"),
        make_artifact("PRIMARY_CODER", 1, "Round 1"),
    ])
    ordered = timeline.ordered()
    assert ordered[0].round_id == 1
    assert ordered[1].round_id == 2


def test_timeline_is_empty():
    assert ArtifactTimeline([]).is_empty
    assert not ArtifactTimeline(build_artifacts(1)).is_empty


def test_timeline_for_role():
    artifacts = build_artifacts(3) + [make_artifact("JUDGE", 10)]
    timeline = ArtifactTimeline(artifacts)
    assert all(a.role == "JUDGE" for a in timeline.for_role("JUDGE"))
    assert len(timeline.for_role("JUDGE")) == 1


def test_timeline_for_round():
    timeline = ArtifactTimeline(build_artifacts(5))
    assert all(a.round_id == 2 for a in timeline.for_round(2))


def test_timeline_latest():
    timeline = ArtifactTimeline(build_artifacts(10))
    latest = timeline.latest(n=3)
    assert [a.round_id for a in latest] == [7, 8, 9]


def test_timeline_grouped_by_round():
    timeline = ArtifactTimeline(build_artifacts(4))
    groups = timeline.grouped_by_round()
    assert set(groups.keys()) == {0, 1, 2, 3}


def test_timeline_contains():
    timeline = ArtifactTimeline(build_artifacts(3))
    assert "PRIMARY_CODER-0" in timeline
    assert "ghost" not in timeline


# ── ArtifactReplayEngine ──────────────────────────────────────────────────────

def test_replay_policy_limit():
    engine = ArtifactReplayEngine(TimelinePolicy(max_replay_artifacts=3, preserve_recent=0))
    replay = engine.replay(ArtifactTimeline(build_artifacts(10)))
    assert len(replay) == 3
    assert replay[0].round_id == 7
    assert replay[-1].round_id == 9


def test_replay_empty_timeline():
    engine = ArtifactReplayEngine(TimelinePolicy.default())
    assert engine.replay(ArtifactTimeline([])) == []


def test_replay_drops_empty_content():
    artifacts = build_artifacts(3)
    artifacts.append(make_artifact("PRIMARY_CODER", 99, content="   "))
    engine = ArtifactReplayEngine(TimelinePolicy(
        max_replay_artifacts=10,
        preserve_recent=0,
        drop_empty_content=True,
    ))
    replay = engine.replay(ArtifactTimeline(artifacts))
    assert all(a.content.strip() for a in replay)


def test_replay_for_role():
    artifacts = build_artifacts(3) + [make_artifact("JUDGE", 10)]
    engine = ArtifactReplayEngine(TimelinePolicy.default())
    replay = engine.replay_for_role(ArtifactTimeline(artifacts), "JUDGE")
    assert all(a.role == "JUDGE" for a in replay)


def test_replay_latest():
    engine = ArtifactReplayEngine(TimelinePolicy.default())
    replay = engine.replay_latest(ArtifactTimeline(build_artifacts(10)), n=2)
    assert len(replay) == 2


def test_replay_priority_roles():
    artifacts = build_artifacts(3) + [make_artifact("JUDGE", 10)]
    engine = ArtifactReplayEngine(TimelinePolicy(
        max_replay_artifacts=10,
        preserve_recent=0,
        priority_roles=["JUDGE"],
        order_by="round_role",
    ))
    replay = engine.replay(ArtifactTimeline(artifacts))
    assert replay[0].role == "JUDGE"


def test_replay_char_budget():
    artifacts = [make_artifact("PRIMARY_CODER", i, "x" * 1000) for i in range(10)]
    engine = ArtifactReplayEngine(TimelinePolicy(
        max_replay_artifacts=10,
        preserve_recent=0,
        max_content_chars=2500,
    ))
    replay = engine.replay(ArtifactTimeline(artifacts))
    assert sum(len(a.content) for a in replay) <= 2500


# ── ReplayContextBuilder ──────────────────────────────────────────────────────

def test_replay_context_builder():
    builder = ReplayContextBuilder()
    context = builder.build(build_artifacts(2))
    assert "[REPLAY ROLE=" in context
    assert "Artifact 0" in context
    assert "Artifact 1" in context


def test_replay_context_empty():
    assert ReplayContextBuilder().build([]) == ""


def test_replay_context_skips_empty_content():
    artifacts = [
        make_artifact("PRIMARY_CODER", 0, "real content"),
        make_artifact("PRIMARY_CODER", 1, "   "),
    ]
    context = ReplayContextBuilder().build(artifacts)
    assert "real content" in context
    assert context.count("[REPLAY ROLE=") == 1


def test_replay_context_ordered_by_round():
    artifacts = [
        make_artifact("PRIMARY_CODER", 2, "second"),
        make_artifact("PRIMARY_CODER", 0, "first"),
    ]
    context = ReplayContextBuilder().build(artifacts)
    assert context.index("first") < context.index("second")


def test_replay_context_build_for_role():
    artifacts = build_artifacts(2) + [make_artifact("JUDGE", 10)]
    context = ReplayContextBuilder().build_for_role(artifacts, "JUDGE")
    assert "ROLE=JUDGE" in context
    assert "ROLE=PRIMARY_CODER" not in context


def test_replay_context_build_latest_round():
    artifacts = build_artifacts(3)
    context = ReplayContextBuilder().build_latest_round(artifacts)
    assert "Artifact 2" in context
    assert "Artifact 0" not in context


# ── TimelinePolicy ────────────────────────────────────────────────────────────

def test_policy_invalid_max_replay():
    with pytest.raises(ValueError):
        TimelinePolicy(max_replay_artifacts=0)


def test_policy_preserve_recent_exceeds_max():
    with pytest.raises(ValueError):
        TimelinePolicy(max_replay_artifacts=3, preserve_recent=5)


def test_policy_invalid_order_by():
    with pytest.raises(ValueError):
        TimelinePolicy(order_by="invalid")


def test_policy_presets_instantiate():
    assert TimelinePolicy.default()
    assert TimelinePolicy.tight()
    assert TimelinePolicy.judge_focused()
    assert TimelinePolicy.full_replay()