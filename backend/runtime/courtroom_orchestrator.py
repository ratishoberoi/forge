from __future__ import annotations
from backend.runtime.courtroom_artifact import CourtroomArtifact
from backend.runtime.courtroom_review import CourtroomReview
from backend.runtime.courtroom_round import CourtroomRound


class CourtroomOrchestrationError(Exception):
    """Raised when orchestration rules are violated."""


class CourtroomOrchestrator:
    """
    Orchestrates sequential courtroom cognition rounds.
    Responsibilities:
    - attach reviews to rounds
    - propagate critiques to artifact
    - apply revisions from coder
    - enforce acceptance rules
    - summarize round state
    """

    def add_review(
        self,
        *,
        round_state: CourtroomRound,
        review: CourtroomReview,
    ) -> None:
        """
        Attach a review to the round and propagate critique to artifact.
        Raises if round is already accepted.
        """
        if round_state.accepted:
            raise CourtroomOrchestrationError(
                f"Cannot add review to accepted round '{round_state.round_id}'."
            )
        round_state.reviews.append(review)
        round_state.artifact.add_critique(review.critique)

    def add_revision(
        self,
        *,
        round_state: CourtroomRound,
        revised_patch: str,
    ) -> None:
        """
        Apply a revised patch to the artifact.
        Raises if round is already accepted.
        """
        if round_state.accepted:
            raise CourtroomOrchestrationError(
                f"Cannot revise accepted round '{round_state.round_id}'."
            )
        round_state.artifact.add_revision(revised_patch)

    def mark_accepted(self, round_state: CourtroomRound) -> None:
        """
        Mark round as accepted.
        Raises if blocking reviews are present.
        """
        if round_state.has_blocking_reviews:
            blocking = round_state.blocking_reviews
            roles = [r.reviewer_role for r in blocking]
            raise CourtroomOrchestrationError(
                f"Cannot accept round '{round_state.round_id}' — "
                f"blocking reviews from: {roles}"
            )
        round_state.accepted = True

    def force_accept(self, round_state: CourtroomRound) -> None:
        """
        Force acceptance regardless of blocking reviews.
        Use only when human override is explicitly intended.
        """
        round_state.accepted = True

    def summarize(self, round_state: CourtroomRound) -> str:
        """Return a human-readable summary of the round state."""
        lines = [
            f"Round:     {round_state.round_id}",
            f"Objective: {round_state.artifact.objective}",
            f"Accepted:  {round_state.accepted}",
            f"Reviews:   {round_state.review_count}",
            f"Blocking:  {round_state.has_blocking_reviews}",
            f"Severity:  {round_state.severity_summary}",
            f"Revisions: {round_state.artifact.revision_count}",
        ]
        return "\n".join(lines)