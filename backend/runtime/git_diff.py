from __future__ import annotations
import subprocess
from pathlib import Path


class GitDiffError(Exception):
    """Raised when git diff operation fails."""


class GitDiff:
    """
    Reads repository diff state via git subprocess.
    Responsibilities:
    - return unstaged diff (working tree vs index)
    - return staged diff (index vs HEAD)
    - return full diff (working tree vs HEAD)
    - support scoped diff for specific paths
    - check if repo is dirty
    """

    def __init__(self, repo_root: str | None = None) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else None

    def diff(self, path: str | None = None) -> str:
        """
        Return unstaged diff (working tree vs index).
        Optionally scoped to a specific path.
        """
        cmd = ["git", "diff"]
        if path:
            cmd += ["--", path]
        return self._run(cmd)

    def diff_staged(self, path: str | None = None) -> str:
        """Return staged diff (index vs HEAD)."""
        cmd = ["git", "diff", "--cached"]
        if path:
            cmd += ["--", path]
        return self._run(cmd)

    def diff_head(self, path: str | None = None) -> str:
        """Return full diff (working tree vs HEAD)."""
        cmd = ["git", "diff", "HEAD"]
        if path:
            cmd += ["--", path]
        return self._run(cmd)

    def is_dirty(self) -> bool:
        """Return True if working tree has any uncommitted changes."""
        return bool(self.diff().strip()) or bool(self.diff_staged().strip())

    def changed_files(self) -> list[str]:
        """Return list of files with unstaged changes."""
        result = self._run(["git", "diff", "--name-only"])
        return [f for f in result.splitlines() if f.strip()]

    def stat(self) -> str:
        """Return diffstat summary — insertions/deletions per file."""
        return self._run(["git", "diff", "--stat"])

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self, cmd: list[str]) -> str:
        """Run git command in repo_root. Raises GitDiffError on failure."""
        kwargs: dict = {
            "capture_output": True,
            "text": True,
        }
        if self.repo_root:
            kwargs["cwd"] = str(self.repo_root)

        try:
            result = subprocess.run(cmd, **kwargs)
            if result.returncode != 0:
                raise GitDiffError(
                    f"git command {cmd} failed: {result.stderr.strip()}"
                )
            return result.stdout
        except FileNotFoundError as exc:
            raise GitDiffError(
                "git executable not found. Is git installed?"
            ) from exc
        except OSError as exc:
            raise GitDiffError(
                f"git command {cmd} failed with OS error: {exc}"
            ) from exc