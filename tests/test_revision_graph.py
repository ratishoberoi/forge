import pytest
from datetime import datetime, timezone
from backend.runtime.artifact_revision import ArtifactRevision
from backend.runtime.revision_branch import RevisionBranch
from backend.runtime.revision_graph import RevisionGraph
from backend.runtime.revision_merge import RevisionMerge
from backend.runtime.revision_tracker import RevisionTracker, RevisionTrackerError


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_revision(
    revision_id: str,
    parent: str | None = None,
    role: str = "coder",
    artifact_id: str = "artifact_1",
) -> ArtifactRevision:
    return ArtifactRevision(
        revision_id=revision_id,
        artifact_id=artifact_id,
        parent_revision_id=parent,
        role=role,
        created_at=datetime.now(timezone.utc),
        summary="revision",
    )


def make_tracker() -> tuple[RevisionTracker, RevisionGraph]:
    graph = RevisionGraph()
    tracker = RevisionTracker(graph)
    return tracker, graph


# ── RevisionGraph ─────────────────────────────────────────────────────────────

def test_graph_add_revision():
    graph = RevisionGraph()
    r1 = build_revision("r1")
    graph.add_revision(r1)
    assert graph.contains("r1")
    assert len(graph) == 1


def test_graph_children():
    graph = RevisionGraph()
    graph.add_revision(build_revision("r1"))
    graph.add_revision(build_revision("r2", parent="r1"))
    assert "r2" in graph.children("r1")


def test_graph_children_empty_for_leaf():
    graph = RevisionGraph()
    graph.add_revision(build_revision("r1"))
    assert graph.children("r1") == set()


def test_graph_parents():
    graph = RevisionGraph()
    graph.add_revision(build_revision("r1"))
    graph.add_revision(build_revision("r2", parent="r1"))
    assert "r1" in graph.parents("r2")


def test_graph_roots():
    graph = RevisionGraph()
    graph.add_revision(build_revision("r1"))
    graph.add_revision(build_revision("r2", parent="r1"))
    roots = graph.roots()
    assert len(roots) == 1
    assert roots[0].revision_id == "r1"


def test_graph_leaves():
    graph = RevisionGraph()
    graph.add_revision(build_revision("r1"))
    graph.add_revision(build_revision("r2", parent="r1"))
    graph.add_revision(build_revision("r3", parent="r2"))
    leaves = graph.leaves()
    assert len(leaves) == 1
    assert leaves[0].revision_id == "r3"


def test_graph_ancestors():
    graph = RevisionGraph()
    graph.add_revision(build_revision("r1"))
    graph.add_revision(build_revision("r2", parent="r1"))
    graph.add_revision(build_revision("r3", parent="r2"))
    ancestors = graph.ancestors("r3")
    assert "r1" in ancestors
    assert "r2" in ancestors


def test_graph_lineage():
    graph = RevisionGraph()
    graph.add_revision(build_revision("r1"))
    graph.add_revision(build_revision("r2", parent="r1"))
    graph.add_revision(build_revision("r3", parent="r2"))
    lineage = graph.lineage("r3")
    assert [r.revision_id for r in lineage] == ["r1", "r2", "r3"]


def test_graph_branching():
    graph = RevisionGraph()
    graph.add_revision(build_revision("r1"))
    graph.add_revision(build_revision("r2a", parent="r1"))
    graph.add_revision(build_revision("r2b", parent="r1"))
    assert "r2a" in graph.children("r1")
    assert "r2b" in graph.children("r1")


# ── RevisionTracker ───────────────────────────────────────────────────────────

def test_tracker_track_and_descendants():
    tracker, _ = make_tracker()
    tracker.track(build_revision("r1"))
    tracker.track(build_revision("r2", parent="r1"))
    tracker.track(build_revision("r3", parent="r2"))

    descendants = tracker.descendants("r1")
    assert "r2" in descendants
    assert "r3" in descendants


def test_tracker_descendants_empty_for_leaf():
    tracker, _ = make_tracker()
    tracker.track(build_revision("r1"))
    assert tracker.descendants("r1") == set()


def test_tracker_duplicate_raises():
    tracker, _ = make_tracker()
    tracker.track(build_revision("r1"))
    with pytest.raises(RevisionTrackerError, match="already tracked"):
        tracker.track(build_revision("r1"))


def test_tracker_descendants_unknown_raises():
    tracker, _ = make_tracker()
    with pytest.raises(RevisionTrackerError, match="not found"):
        tracker.descendants("nonexistent")


def test_tracker_ancestors():
    tracker, _ = make_tracker()
    tracker.track(build_revision("r1"))
    tracker.track(build_revision("r2", parent="r1"))
    tracker.track(build_revision("r3", parent="r2"))

    ancestors = tracker.ancestors("r3")
    assert "r1" in ancestors
    assert "r2" in ancestors


def test_tracker_lineage():
    tracker, _ = make_tracker()
    tracker.track(build_revision("r1"))
    tracker.track(build_revision("r2", parent="r1"))
    tracker.track(build_revision("r3", parent="r2"))

    lineage = tracker.lineage("r3")
    assert [r.revision_id for r in lineage] == ["r1", "r2", "r3"]


def test_tracker_latest_for_artifact():
    tracker, _ = make_tracker()
    tracker.track(build_revision("r1", artifact_id="a1"))
    tracker.track(build_revision("r2", parent="r1", artifact_id="a1"))

    latest = tracker.latest_for_artifact("a1")
    assert latest.revision_id == "r2"


def test_tracker_latest_for_artifact_none():
    tracker, _ = make_tracker()
    assert tracker.latest_for_artifact("nonexistent") is None


def test_tracker_revisions_by_role():
    tracker, _ = make_tracker()
    tracker.track(build_revision("r1", role="coder"))
    tracker.track(build_revision("r2", parent="r1", role="judge"))
    tracker.track(build_revision("r3", parent="r2", role="coder"))

    coder_revs = tracker.revisions_by_role("coder")
    assert len(coder_revs) == 2
    assert all(r.role == "coder" for r in coder_revs)


def test_tracker_is_branched():
    tracker, _ = make_tracker()
    tracker.track(build_revision("r1"))
    tracker.track(build_revision("r2a", parent="r1"))
    tracker.track(build_revision("r2b", parent="r1"))

    assert tracker.is_branched("r1") is True
    assert tracker.is_branched("r2a") is False


def test_tracker_roots_and_leaves():
    tracker, _ = make_tracker()
    tracker.track(build_revision("r1"))
    tracker.track(build_revision("r2", parent="r1"))

    assert len(tracker.roots) == 1
    assert tracker.roots[0].revision_id == "r1"
    assert len(tracker.leaves) == 1
    assert tracker.leaves[0].revision_id == "r2"


def test_tracker_revision_count():
    tracker, _ = make_tracker()
    tracker.track(build_revision("r1"))
    tracker.track(build_revision("r2", parent="r1"))
    assert tracker.revision_count == 2


# ── RevisionMerge ─────────────────────────────────────────────────────────────

def test_merge_creation():
    merge = RevisionMerge(
        source_revision="r2a",
        target_revision="r2b",
        merged_revision="r3",
    )
    assert merge.source_revision == "r2a"
    assert merge.target_revision == "r2b"
    assert merge.merged_revision == "r3"


def test_merge_same_source_target_raises():
    with pytest.raises(ValueError, match="differ"):
        RevisionMerge(
            source_revision="r1",
            target_revision="r1",
            merged_revision="r2",
        )


def test_merge_factory():
    merge = RevisionMerge.create(
        source_revision="r2a",
        target_revision="r2b",
    )
    assert merge.merged_revision != ""
    assert merge.source_revision != merge.target_revision


def test_merge_involves():
    merge = RevisionMerge(
        source_revision="r2a",
        target_revision="r2b",
        merged_revision="r3",
    )
    assert "r2a" in merge.involves
    assert "r2b" in merge.involves
    assert "r3" in merge.involves


def test_merge_to_dict():
    merge = RevisionMerge.create(source_revision="r2a", target_revision="r2b")
    d = merge.to_dict()
    assert d["source_revision"] == "r2a"
    assert "created_at" in d


# ── RevisionBranch ────────────────────────────────────────────────────────────

def test_branch_creation():
    branch = RevisionBranch(branch_id="main")
    assert branch.is_empty is True
    assert branch.tip is None
    assert branch.root is None


def test_branch_append():
    branch = RevisionBranch(branch_id="main")
    branch.append("r1")
    branch.append("r2")
    assert branch.tip == "r2"
    assert branch.root == "r1"
    assert branch.length == 2


def test_branch_contains():
    branch = RevisionBranch(branch_id="main")
    branch.append("r1")
    assert branch.contains("r1") is True
    assert branch.contains("r99") is False


def test_branch_duplicate_raises():
    branch = RevisionBranch(branch_id="main")
    branch.append("r1")
    with pytest.raises(ValueError, match="already exists"):
        branch.append("r1")


def test_branch_empty_id_raises():
    with pytest.raises(ValueError, match="branch_id"):
        RevisionBranch(branch_id="")


def test_branch_to_dict():
    branch = RevisionBranch(branch_id="main")
    branch.append("r1")
    d = branch.to_dict()
    assert d["branch_id"] == "main"
    assert "r1" in d["revisions"]
    assert "created_at" in d