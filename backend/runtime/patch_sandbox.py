from __future__ import annotations
from pathlib import Path
from backend.runtime.file_editor import FileEditor
from backend.runtime.output_parser import ParsedPatchOutput


class PatchSandbox:
    """
    Safe patch materialization sandbox.
    Responsibilities:
    - isolate generated files
    - prevent path traversal
    - materialize cognition outputs
    - prepare filesystem state for diffing
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.editor = FileEditor(str(self.root))

    def _safe_resolve(self, relative_path: str) -> Path:
        """
        Resolve relative_path under sandbox root.
        Raises ValueError on path traversal attempt.
        """
        resolved = (self.root / relative_path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(
                f"Path traversal detected: '{relative_path}' escapes sandbox root."
            )
        return resolved

    async def materialize_patch(self, parsed: ParsedPatchOutput) -> list[Path]:
        """Write all files from parsed output into sandbox. Returns resolved paths."""
        if not parsed.files:
            raise ValueError("ParsedPatchOutput.files is empty — nothing to materialize.")

        written: list[Path] = []

        for relative_path, content in parsed.files.items():
            resolved = self._safe_resolve(relative_path)
            self.editor.write_file(relative_path, content)
            written.append(resolved)

        return written

    def list_materialized(self) -> list[Path]:
        """Return all files currently present in the sandbox root."""
        return sorted(self.root.rglob("*"))

    def is_within_sandbox(self, path: str | Path) -> bool:
        """Check whether a given path is safely inside the sandbox root."""
        try:
            Path(path).resolve().relative_to(self.root)
            return True
        except ValueError:
            return False