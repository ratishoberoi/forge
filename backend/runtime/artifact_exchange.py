from __future__ import annotations
from pathlib import Path
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_loader import ArtifactLoader
from backend.runtime.artifact_store import ArtifactStore


class ArtifactExchangeError(Exception):
    """Raised when ArtifactExchange encounters a non-recoverable error."""


class ArtifactExchange:
    """
    Runtime-to-runtime cognition exchange layer.
    Coordinates:
    - persistence of cognition artifacts across runtime swaps
    - retrieval by role and round_id
    - full role history for context building
    Pattern:
        coder runtime  → persist()             → disk
        judge runtime  → retrieve_round()      → disk
        any runtime    → retrieve_role_history → disk
    """

    def __init__(
        self,
        *,
        store: ArtifactStore,
        loader: ArtifactLoader,
    ) -> None:
        self.store = store
        self.loader = loader
        self._persist_log: list[str] = []

    # ── Write ─────────────────────────────────────────────────────────────────

    def persist(self, artifact: CognitionArtifact) -> Path:
        """
        Persist a CognitionArtifact to disk.
        Returns the path written — pass to next runtime as handoff reference.
        """
        if not artifact.content.strip():
            raise ArtifactExchangeError(
                f"Artifact '{artifact.artifact_id}' has blank content — "
                "will not persist."
            )
        path = self.store.save(artifact)
        self._persist_log.append(artifact.artifact_id)
        return path

    # ── Read ──────────────────────────────────────────────────────────────────

    def retrieve_round(
        self,
        *,
        role: str,
        round_id: int,
    ) -> CognitionArtifact:
        """
        Retrieve a single artifact by role and round_id.
        Raises ArtifactExchangeError if not found.
        """
        try:
            return self.loader.load_round(role, round_id)
        except Exception as exc:
            raise ArtifactExchangeError(
                f"Failed to retrieve artifact: role='{role}' "
                f"round_id={round_id}: {exc}"
            ) from exc

    def retrieve_role_history(
        self,
        role: str,
    ) -> list[CognitionArtifact]:
        """
        Retrieve full artifact history for a role, sorted by round_id ascending.
        Returns empty list if no artifacts exist for the role.
        """
        return self.loader.load_role_artifacts(role)

    def retrieve_latest(self, role: str) -> CognitionArtifact | None:
        """
        Retrieve the most recent artifact for a role.
        Returns None if no artifacts exist.
        """
        history = self.retrieve_role_history(role)
        return history[-1] if history else None

    def exists(self, role: str, round_id: int) -> bool:
        """Return True if an artifact exists for the given role and round_id."""
        return self.loader.exists(role, round_id)

    # ── Audit ─────────────────────────────────────────────────────────────────

    @property
    def persist_count(self) -> int:
        return len(self._persist_log)

    @property
    def persisted_ids(self) -> list[str]:
        return list(self._persist_log)