from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from backend.runtime.artifact import CognitionArtifact


class ArtifactLoaderError(Exception):
    """Raised when ArtifactLoader encounters a non-recoverable error."""


class ArtifactLoader:
    """
    Read-only artifact loader.
    Complements ArtifactStore — no write operations.
    Responsibilities:
    - load all artifacts for a given role
    - load a single artifact by role and round_id
    - filter artifacts by metadata or round range
    - validate deserialized structure
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir).resolve()

    # ── Core load ────────────────────────────────────────────────────────────

    def load_role_artifacts(self, role: str) -> list[CognitionArtifact]:
        """Load all artifacts for a role, sorted by round_id ascending."""
        role_dir = self._role_dir(role)
        if not role_dir.exists():
            return []

        artifacts = [
            self._deserialize(path)
            for path in sorted(role_dir.glob("round_*.json"))
        ]
        return sorted(artifacts, key=lambda a: a.round_id)

    def load_round(self, role: str, round_id: int) -> CognitionArtifact:
        """Load a single artifact by role and round_id. Raises if not found."""
        path = self._role_dir(role) / f"round_{round_id}.json"
        if not path.exists():
            raise ArtifactLoaderError(
                f"Artifact not found: role='{role}' round_id={round_id} at {path}"
            )
        return self._deserialize(path)

    def exists(self, role: str, round_id: int) -> bool:
        """Return True if artifact file exists for given role and round_id."""
        return (self._role_dir(role) / f"round_{round_id}.json").exists()

    # ── Filtered load ────────────────────────────────────────────────────────

    def load_round_range(
        self,
        role: str,
        *,
        start: int,
        end: int,
    ) -> list[CognitionArtifact]:
        """Load artifacts for a role within [start, end] round_id range inclusive."""
        return [
            a for a in self.load_role_artifacts(role)
            if start <= a.round_id <= end
        ]

    def load_by_metadata(
        self,
        role: str,
        **filters: object,
    ) -> list[CognitionArtifact]:
        """
        Load artifacts whose metadata contains all given key=value pairs.
        Example: loader.load_by_metadata("coder", agent_id="coder-agent")
        """
        return [
            a for a in self.load_role_artifacts(role)
            if all(a.metadata.get(k) == v for k, v in filters.items())
        ]

    def list_roles(self) -> list[str]:
        """Return all role names that have artifact directories."""
        if not self.base_dir.exists():
            return []
        return sorted(
            p.name for p in self.base_dir.iterdir()
            if p.is_dir()
        )

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
            raise ArtifactLoaderError(
                f"Failed to deserialize artifact from {path}: {exc}"
            ) from exc