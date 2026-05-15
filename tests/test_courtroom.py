import pytest
from backend.runtime.courtroom_artifact import CourtroomArtifact
from backend.runtime.courtroom_orchestrator import (
    CourtroomOrchestrator,
    CourtroomOrchestrationError,
)
from backend.runtime.courtroom_review import CourtroomReview
from backend.runtime.courtroom_round import CourtroomRound


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_artifact(
    patch: str = "patch_v1",
    objective: str = "Improve auth flow",
) -> CourtroomArtifact:
    return CourtroomArtifact(
        artifact_id="a1",
        objective=objective,
        patch=patch,
    )


def build_round(artifact: CourtroomArtifact | None = None) -> CourtroomRound:
    return CourtroomRound(
        round_id="r1",
        artifact=artifact or build_artifact(),
    )


def build_review(
    severity: str = "high",
    critique: str = "Auth dependency risk",
    role: str = "DEEPSEEK_SYNTH",
) -> CourtroomReview:
    return CourtroomReview(
        reviewer_role=role,
        critique=critique,
        severity=severity,
    )


def make_orchestrator() -> CourtroomOrchestrator:
    return CourtroomOrchestrator()


# ── CourtroomArtifact ─────────────────────────────────────────────────────────

def test_artifact_creation():
    artifact = build_artifact()
    assert artifact.artifact_id == "a1"
    assert artifact.objective == "Improve auth flow"
    assert artifact.patch == "patch_v1"
    assert artifact.critiques == []
    assert artifact.revisions == []


def test_artifact_factory():
    artifact = CourtroomArtifact.create(
        objective="Add typing",
        patch="def hello(name: str): ...",
    )
    assert artifact.artifact_id != ""
    assert artifact.objective == "Add typing"


def test_artifact_add_critique():
    artifact = build_artifact()
    artifact.add_critique("Risk identified.")
    assert "Risk identified." in artifact.critiques
    assert artifact.critique_count == 1


def test_artifact_add_revision():
    artifact = build_artifact()
    artifact.add_revision("patch_v2")
    assert artifact.latest_revision == "patch_v2"
    assert artifact.revision_count == 1


def test_artifact_blank_critique_raises():
    artifact = build_artifact()
    with pytest.raises(ValueError, match="blank"):
        artifact.add_critique("   ")


def test_artifact_empty_patch_raises():
    with pytest.raises(ValueError, match="patch"):
        CourtroomArtifact(artifact_id="a1", objective="obj", patch="")


def test_artifact_to_dict():
    artifact = build_artifact()
    d = artifact.to_dict()
    assert d["artifact_id"] == "a1"
    assert d["patch"] == "patch_v1"
    assert "created_at" in d


# ── CourtroomReview ───────────────────────────────────────────────────────────

def test_review_creation():
    review = build_review()
    assert review.reviewer_role == "DEEPSEEK_SYNTH"
    assert review.critique == "Auth dependency risk"
    assert review.severity == "high"


def test_review_is_blocking_high():
    review = build_review(severity="high")
    assert review.is_blocking is True


def test_review_is_blocking_critical():
    review = build_review(severity="critical")
    assert review.is_blocking is True
    assert review.is_critical is True


def test_review_not_blocking_low():
    review = build_review(severity="low")
    assert review.is_blocking is False


def test_review_invalid_severity_raises():
    with pytest.raises(ValueError, match="severity"):
        CourtroomReview(
            reviewer_role="JUDGE",
            critique="Some critique",
            severity="extreme",
        )


def test_review_blank_critique_raises():
    with pytest.raises(ValueError, match="blank"):
        CourtroomReview(
            reviewer_role="JUDGE",
            critique="   ",
            severity="low",
        )


def test_review_factory():
    review = CourtroomReview.create(
        reviewer_role="JUDGE",
        critique="Looks good.",
        severity="low",
    )
    assert review.review_id != ""


# ── CourtroomRound ────────────────────────────────────────────────────────────

def test_round_creation():
    round_state = build_round()
    assert round_state.round_id == "r1"
    assert round_state.accepted is False
    assert round_state.reviews == []


def test_round_factory():
    artifact = build_artifact()
    round_state = CourtroomRound.create(artifact=artifact)
    assert round_state.round_id != ""


def test_round_has_blocking_reviews():
    round_state = build_round()
    round_state.reviews.append(build_review(severity="critical"))
    assert round_state.has_blocking_reviews is True


def test_round_severity_summary():
    round_state = build_round()
    round_state.reviews.append(build_review(severity="low"))
    round_state.reviews.append(build_review(severity="high"))
    round_state.reviews.append(build_review(severity="high"))
    assert round_state.severity_summary == {"low": 1, "high": 2}


def test_round_reviews_by_role():
    round_state = build_round()
    round_state.reviews.append(build_review(role="DEEPSEEK_SYNTH", severity="high"))
    round_state.reviews.append(build_review(role="JUDGE", severity="low"))
    by_role = round_state.reviews_by_role
    assert "DEEPSEEK_SYNTH" in by_role
    assert "JUDGE" in by_role


def test_round_to_dict():
    round_state = build_round()
    d = round_state.to_dict()
    assert d["round_id"] == "r1"
    assert "artifact" in d
    assert "reviews" in d


# ── CourtroomOrchestrator ─────────────────────────────────────────────────────

def test_add_review():
    orchestrator = make_orchestrator()
    round_state = build_round()
    review = build_review()

    orchestrator.add_review(round_state=round_state, review=review)

    assert len(round_state.reviews) == 1
    assert "Auth dependency risk" in round_state.artifact.critiques


def test_add_multiple_reviews():
    orchestrator = make_orchestrator()
    round_state = build_round()

    orchestrator.add_review(round_state=round_state, review=build_review(role="DEEPSEEK_SYNTH", severity="high"))
    orchestrator.add_review(round_state=round_state, review=build_review(role="JUDGE", severity="low"))

    assert round_state.review_count == 2
    assert len(round_state.artifact.critiques) == 2


def test_add_review_to_accepted_round_raises():
    orchestrator = make_orchestrator()
    round_state = build_round()
    round_state.accepted = True

    with pytest.raises(CourtroomOrchestrationError, match="accepted"):
        orchestrator.add_review(round_state=round_state, review=build_review())


def test_add_revision():
    orchestrator = make_orchestrator()
    round_state = build_round()

    orchestrator.add_revision(round_state=round_state, revised_patch="patch_v2")

    assert round_state.artifact.latest_revision == "patch_v2"


def test_add_revision_to_accepted_round_raises():
    orchestrator = make_orchestrator()
    round_state = build_round()
    round_state.accepted = True

    with pytest.raises(CourtroomOrchestrationError, match="accepted"):
        orchestrator.add_revision(round_state=round_state, revised_patch="patch_v2")


def test_mark_accepted():
    orchestrator = make_orchestrator()
    round_state = build_round()

    orchestrator.mark_accepted(round_state)

    assert round_state.accepted is True


def test_mark_accepted_with_blocking_review_raises():
    orchestrator = make_orchestrator()
    round_state = build_round()
    orchestrator.add_review(round_state=round_state, review=build_review(severity="high"))

    with pytest.raises(CourtroomOrchestrationError, match="blocking"):
        orchestrator.mark_accepted(round_state)


def test_mark_accepted_with_low_severity_passes():
    orchestrator = make_orchestrator()
    round_state = build_round()
    orchestrator.add_review(round_state=round_state, review=build_review(severity="low"))

    orchestrator.mark_accepted(round_state)

    assert round_state.accepted is True


def test_force_accept_bypasses_blocking():
    orchestrator = make_orchestrator()
    round_state = build_round()
    orchestrator.add_review(round_state=round_state, review=build_review(severity="critical"))

    orchestrator.force_accept(round_state)

    assert round_state.accepted is True


def test_summarize():
    orchestrator = make_orchestrator()
    round_state = build_round()
    orchestrator.add_review(round_state=round_state, review=build_review(severity="low"))

    summary = orchestrator.summarize(round_state)

    assert "Round:" in summary
    assert "Objective:" in summary
    assert "Accepted:" in summary
    assert "Severity:" in summary