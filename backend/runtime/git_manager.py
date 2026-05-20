from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GitManagerError(RuntimeError):
    """Raised when a git safety operation fails."""


@dataclass(slots=True)
class GitStatusSnapshot:
    branch: str
    is_dirty: bool
    modified_files: list[str]
    untracked_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "is_dirty": self.is_dirty,
            "modified_files": self.modified_files,
            "untracked_files": self.untracked_files,
        }


@dataclass(slots=True)
class CommitInfo:
    sha: str
    subject: str
    author: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "sha": self.sha,
            "subject": self.subject,
            "author": self.author,
            "created_at": self.created_at,
        }


class GitManager:
    PROTECTED_BRANCHES = {"main", "master"}

    def __init__(self, repo_path: str, *, env: dict[str, str] | None = None) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.env = {**os.environ, **(env or {})}
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise GitManagerError(f"Repository path does not exist: {self.repo_path}")

    def is_git_repository(self) -> bool:
        try:
            self._git("rev-parse", "--is-inside-work-tree")
            return True
        except GitManagerError:
            return False

    def current_branch(self) -> str:
        branch = self._git("branch", "--show-current")
        if branch:
            return branch
        return self._git("rev-parse", "--short", "HEAD")

    def head_sha(self) -> str:
        return self._git("rev-parse", "HEAD")

    def status(self) -> GitStatusSnapshot:
        output = self._git("status", "--porcelain")
        modified: list[str] = []
        untracked: list[str] = []
        for line in output.splitlines():
            if not line:
                continue
            state = line[:2]
            path = line[3:] if len(line) > 3 else ""
            if state == "??":
                untracked.append(path)
            else:
                modified.append(path)
        return GitStatusSnapshot(
            branch=self.current_branch(),
            is_dirty=bool(modified or untracked),
            modified_files=modified,
            untracked_files=untracked,
        )

    def create_branch(self, branch_name: str, *, checkout: bool = True) -> str:
        self._validate_branch(branch_name)
        existing = self._git_allow_error("rev-parse", "--verify", branch_name)
        if existing.returncode == 0:
            if checkout:
                self.switch_branch(branch_name)
            return branch_name
        args = ("checkout", "-b", branch_name) if checkout else ("branch", branch_name)
        self._git(*args)
        return branch_name

    def switch_branch(self, branch_name: str) -> str:
        self._validate_branch(branch_name)
        self._git("checkout", branch_name)
        return self.current_branch()

    def create_execution_branch(self, run_id: str) -> str:
        branch_name = f"forge/run-{_safe_ref(run_id)}"
        current = self.current_branch()
        if current == branch_name:
            return branch_name
        return self.create_branch(branch_name, checkout=True)

    def commit_all(self, message: str) -> str | None:
        if not message.strip():
            raise GitManagerError("commit message must not be blank.")
        self._git("add", "--all")
        staged = self._git_allow_error("diff", "--cached", "--quiet")
        if staged.returncode != 0:
            self._git("commit", "-m", message)
            return self.head_sha()
        return None

    def history(self, limit: int = 20) -> list[CommitInfo]:
        limit = max(1, min(limit, 100))
        output = self._git(
            "log",
            f"-{limit}",
            "--pretty=format:%H%x1f%an%x1f%ad%x1f%s",
            "--date=iso-strict",
        )
        commits: list[CommitInfo] = []
        for line in output.splitlines():
            parts = line.split("\x1f", 3)
            if len(parts) == 4:
                commits.append(
                    CommitInfo(
                        sha=parts[0],
                        author=parts[1],
                        created_at=parts[2],
                        subject=parts[3],
                    )
                )
        return commits

    def rollback(self, target: str = "HEAD", *, clean_untracked: bool = False) -> None:
        self._git("reset", "--hard", target)
        if clean_untracked:
            self._git("clean", "-fd")

    def revert(self, commit_sha: str, *, no_commit: bool = False) -> str:
        if not commit_sha.strip():
            raise GitManagerError("commit_sha must not be blank.")
        args = ["revert"]
        if no_commit:
            args.append("--no-commit")
        args.append(commit_sha)
        self._git(*args)
        return self.head_sha()

    def branches(self) -> list[str]:
        output = self._git("branch", "--format=%(refname:short)")
        return [line.strip() for line in output.splitlines() if line.strip()]

    def diff_name_status(self) -> list[dict[str, str]]:
        output = self._git("diff", "--name-status", "HEAD")
        rows: list[dict[str, str]] = []
        for line in output.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                rows.append({"status": parts[0], "path": parts[-1]})
        return rows

    def _git(self, *args: str, check: bool = True) -> str:
        result = self._git_allow_error(*args)
        if check and result.returncode != 0:
            raise GitManagerError(
                f"git {' '.join(args)} failed in {self.repo_path}: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    def _git_allow_error(self, *args: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.repo_path,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GitManagerError(f"git {' '.join(args)} failed: {exc}") from exc

    def _validate_branch(self, branch_name: str) -> None:
        if not branch_name.strip():
            raise GitManagerError("branch name must not be blank.")
        if branch_name in self.PROTECTED_BRANCHES:
            raise GitManagerError(f"Refusing to modify protected branch: {branch_name}")


def _safe_ref(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)[:80]
