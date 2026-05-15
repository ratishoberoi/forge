from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from backend.runtime.artifact import CognitionArtifact


class ArtifactStoreError(Exception):
    """Raised when ArtifactStore encounters a non-recoverable error."""


class ArtifactStore:
    """
    Filesystem-backed artifact persistence.
    Responsibilities:
    - save cognition artifacts to disk
    - load artifacts by role and round
    - list available artifacts
    - delete artifacts
    Layout: base_dir / role / round_{id}.json
    """

    def __init__(self, base_dir: str = "runtime_artifacts") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Write ────────────────────────────────────────────────────────────────

    def save(self, artifact: CognitionArtifact) -> Path:
        """Persist artifact to disk. Returns the path written."""
        role_dir = self._role_dir(artifact.role)
        role_dir.mkdir(parents=True, exist_ok=True)
        path = role_dir / f"round_{artifact.round_id}.json"
        try:
            path.write_text(
                json.dumps(artifact.to_dict(), indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactStoreError(
                f"Failed to write artifact to {path}: {exc}"
            ) from exc
        return path

    # ── Read ─────────────────────────────────────────────────────────────────

    def load(self, role: str, round_id: int) -> CognitionArtifact:
        """Load a single artifact by role and round_id. Raises if not found."""
        path = self._role_dir(role) / f"round_{round_id}.json"
        if not path.exists():
            raise ArtifactStoreError(
                f"Artifact not found: role='{role}' round_id={round_id} at {path}"
            )
        return self._deserialize(path)

    def load_all(self, role: str) -> list[CognitionArtifact]:
        """Load all artifacts for a given role, sorted by round_id ascending."""
        role_dir = self._role_dir(role)
        if not role_dir.exists():
            return []
        artifacts = [
            self._deserialize(p)
            for p in sorted(role_dir.glob("round_*.json"))
        ]
        return sorted(artifacts, key=lambda a: a.round_id)

    def exists(self, role: str, round_id: int) -> bool:
        """Return True if an artifact exists for the given role and round_id."""
        return (self._role_dir(role) / f"round_{round_id}.json").exists()

    # ── Delete ───────────────────────────────────────────────────────────────

    def delete(self, role: str, round_id: int) -> bool:
        """Delete artifact. Returns True if deleted, False if not found."""
        path = self._role_dir(role) / f"round_{round_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    def clear_role(self, role: str) -> int:
        """Delete all artifacts for a role. Returns count of deleted files."""
        role_dir = self._role_dir(role)
        if not role_dir.exists():
            return 0
        count = 0
        for path in role_dir.glob("round_*.json"):
            path.unlink()
            count += 1
        return count

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _role_dir(self, role: str) -> Path:
        return self.base_dir / role.lower()

    def _deserialize(self, path: Path) -> CognitionArtifact:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CognitionArtifact(
                artifact_id=data["artifact_id"],
                role=data["role"],
                round_id=data["round_id"],
                task=data["task"],
                content=data["content"],
                created_at=datetime.fromisoformat(data["created_at"]),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError, OSError) as exc:
            raise ArtifactStoreError(
                f"Failed to deserialize artifact from {path}: {exc}"
            ) from exc