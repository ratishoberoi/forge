from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FileEditResult:
    path: Path
    existed_before: bool
    original_content: str | None
    updated_content: str


class FileEditor:
    """
    Safe repository mutation layer.
    Handles:
    - controlled file writes
    - path safety
    - deterministic repo mutations
    """

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve repository-relative path safely."""
        target = (self.repo_root / relative_path).resolve()
        if self.repo_root not in target.parents and target != self.repo_root:
            raise ValueError("Path escapes repository root.")
        return target

    def read_file(self, relative_path: str) -> str:
        """Read UTF-8 file contents."""
        return self.resolve_path(relative_path).read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> FileEditResult:
        """Safely write UTF-8 file contents."""
        path = self.resolve_path(relative_path)
        existed_before = path.exists()
        original_content = path.read_text(encoding="utf-8") if existed_before else None

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return FileEditResult(
            path=path,
            existed_before=existed_before,
            original_content=original_content,
            updated_content=content,
        )

    def delete_file(self, relative_path: str) -> bool:
        """Delete file if exists. Returns True if deleted."""
        path = self.resolve_path(relative_path)
        if path.exists():
            path.unlink()
            return True
        return False