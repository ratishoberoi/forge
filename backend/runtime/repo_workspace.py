from __future__ import annotations
from pathlib import Path


class WorkspaceError(Exception):
    """Raised when workspace operation fails or path escapes root."""


class RepositoryWorkspace:
    """
    Controlled repository workspace.
    Responsibilities:
    - resolve relative paths safely under repo root
    - prevent path traversal attacks
    - read and write files
    - list workspace contents
    """

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.exists():
            raise WorkspaceError(
                f"Repository root does not exist: {self.repo_root}"
            )

    def resolve(self, relative_path: str) -> Path:
        """
        Resolve relative_path under repo root.
        Raises WorkspaceError on path traversal attempt.
        """
        resolved = (self.repo_root / relative_path).resolve()
        if not resolved.is_relative_to(self.repo_root):
            raise ValueError(
                f"Path traversal detected: '{relative_path}' "
                f"escapes repository root '{self.repo_root}'."
            )
        return resolved

    def read(self, relative_path: str) -> str:
        """Read file content. Raises WorkspaceError if file not found."""
        path = self.resolve(relative_path)
        if not path.exists():
            raise WorkspaceError(
                f"File not found in workspace: '{relative_path}'"
            )
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise WorkspaceError(
                f"Failed to read '{relative_path}': {exc}"
            ) from exc

    def write(self, *, relative_path: str, content: str) -> Path:
        """
        Write content to relative_path under repo root.
        Creates parent directories as needed.
        Returns resolved path written.
        """
        path = self.resolve(relative_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise WorkspaceError(
                f"Failed to write '{relative_path}': {exc}"
            ) from exc
        return path

    def exists(self, relative_path: str) -> bool:
        """Return True if file exists at relative_path."""
        try:
            return self.resolve(relative_path).exists()
        except ValueError:
            return False

    def delete(self, relative_path: str) -> bool:
        """Delete file at relative_path. Returns True if deleted, False if not found."""
        path = self.resolve(relative_path)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_files(self, pattern: str = "**/*") -> list[Path]:
        """Return all files matching glob pattern under repo root."""
        return sorted(
            p for p in self.repo_root.glob(pattern) if p.is_file()
        )

    def relative_to_root(self, absolute_path: Path) -> str:
        """Convert absolute path back to relative string under repo root."""
        return str(absolute_path.relative_to(self.repo_root))