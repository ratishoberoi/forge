from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

_GIT_TIMEOUT = 30


@dataclass(slots=True)
class RepoStatus:
    branch: str
    modified_files: list[str]
    untracked_files: list[str]


async def _run_git(
    *args: str,
    cwd: Path,
    timeout: float = _GIT_TIMEOUT,
) -> str:
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )

    except asyncio.TimeoutError:
        process.kill()

        try:
            await process.communicate()
        except Exception:
            pass

        raise RuntimeError(
            f"git {' '.join(args)} timed out "
            f"after {timeout}s"
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n"
            f"{stderr.decode().strip()}"
        )

    return stdout.decode().strip()


class GitOperations:
    """
    Centralized git interaction layer.

    Responsible for:
    - diff extraction
    - repository status
    - changed file tracking
    - branch introspection
    """

    def __init__(
        self,
        repo_path: str,
    ) -> None:
        self.repo_path = Path(
            repo_path
        ).resolve()

    async def current_branch(self) -> str:
        return await _run_git(
            "branch",
            "--show-current",
            cwd=self.repo_path,
        )

    async def get_diff(self) -> str:
        """
        Return unified git diff.
        """

        return await _run_git(
            "diff",
            cwd=self.repo_path,
        )
    
    async def stage_all(self) -> None:
        """Stage all changes including new files."""
        await _run_git("add", "--all", cwd=self.repo_path)

    async def get_staged_diff(self) -> str:
        staged = await _run_git("diff", "--cached", cwd=self.repo_path)
        unstaged = await _run_git("diff", cwd=self.repo_path)
        return staged or unstaged

    async def changed_files(
        self,
    ) -> list[str]:
        """
        Return changed tracked files.
        """

        output = await _run_git(
            "diff",
            "--name-only",
            cwd=self.repo_path,
        )

        if not output:
            return []

        return output.splitlines()

    async def untracked_files(
        self,
    ) -> list[str]:
        """
        Return untracked files.
        """

        output = await _run_git(
            "ls-files",
            "--others",
            "--exclude-standard",
            cwd=self.repo_path,
        )

        if not output:
            return []

        return output.splitlines()

    async def repo_status(
        self,
    ) -> RepoStatus:
        """
        Return structured repository status.
        """

        output = await _run_git(
            "status",
            "--porcelain",
            cwd=self.repo_path,
        )

        modified: list[str] = []
        untracked: list[str] = []

        for line in output.splitlines():

            status = line[:2]

            parts = line.split(
                maxsplit=1
            )

            if len(parts) < 2:
                continue

            filename = parts[1]

            if status == "??":
                untracked.append(filename)

            elif "M" in status:
                modified.append(filename)

        return RepoStatus(
            branch=await self.current_branch(),
            modified_files=modified,
            untracked_files=untracked,
        )