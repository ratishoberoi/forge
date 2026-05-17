from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from backend.runtime.repo_workspace import RepositoryWorkspace


class PatchWriterError(Exception):
    """Raised when PatchWriter fails to apply a patch."""


@dataclass(slots=True)
class PatchResult:
    """Result of a single patch apply operation."""
    file_path: str
    success: bool
    resolved_path: Path | None = None
    error: str | None = None


class PatchWriter:
    """
    Applies generated patches into the repository workspace.
    Responsibilities:
    - write new content to target files
    - optionally backup originals before overwrite
    - apply multiple files in one operation
    - report per-file results
    """

    def __init__(
        self,
        workspace: RepositoryWorkspace,
        backup: bool = False,
    ) -> None:
        self.workspace = workspace
        self.backup = backup

    def apply(
        self,
        *,
        file_path: str,
        new_content: str,
    ) -> PatchResult:
        """
        Apply new_content to file_path in workspace.
        Optionally backs up original before overwrite.
        Returns PatchResult.
        """
        if not new_content.strip():
            raise PatchWriterError(
                f"Refusing to write blank content to '{file_path}'."
            )

        try:
            if self.backup and self.workspace.exists(file_path):
                original = self.workspace.read(file_path)
                self.workspace.write(
                    relative_path=f"{file_path}.bak",
                    content=original,
                )

            resolved = self.workspace.write(
                relative_path=file_path,
                content=new_content,
            )
            return PatchResult(
                file_path=file_path,
                success=True,
                resolved_path=resolved,
            )

        except Exception as exc:
            return PatchResult(
                file_path=file_path,
                success=False,
                error=str(exc),
            )

    def apply_many(
        self,
        patches: dict[str, str],
    ) -> list[PatchResult]:
        """
        Apply multiple file patches in one operation.
        patches: dict of relative_path → new_content.
        Returns list of PatchResult per file.
        """
        if not patches:
            raise PatchWriterError("patches dict must not be empty.")

        results = []
        for file_path, content in patches.items():
            result = self.apply(file_path=file_path, new_content=content)
            results.append(result)
        return results

    @property
    def failed_results(self) -> list[PatchResult]:
        """Filter helper — not stateful, use with result list from apply_many."""
        return []