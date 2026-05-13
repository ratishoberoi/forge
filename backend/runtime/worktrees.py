from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

_GIT_TIMEOUT = 30


@dataclass(slots=True)
class WorktreeInfo:
    workspace_id: str
    branch_name: str
    path: Path


async def _run_git(
    *args: str,
    cwd: Path,
    timeout: float = _GIT_TIMEOUT,
) -> str:
    """
    Execute a git command safely.

    Raises:
        RuntimeError:
            On timeout or non-zero exit.
    """

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
            f"git {' '.join(args)} timed out after {timeout}s"
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n"
            f"{stderr.decode().strip()}"
        )

    return stdout.decode().strip()


class WorktreeManager:
    """
    Manages isolated git worktrees for autonomous agents.

    Each agent receives:
    - isolated branch
    - isolated filesystem workspace
    - independent diff lifecycle
    """

    def __init__(
        self,
        repo_root: str,
        worktree_root: str = "/home/ratish/ForgeWorktrees",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.worktree_root = Path(worktree_root).resolve()

        self.worktree_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    async def create_worktree(
        self,
        agent_name: str,
    ) -> WorktreeInfo:
        """
        Create isolated git worktree + branch.
        """

        workspace_id = str(uuid.uuid4())[:8]

        branch_name = f"{agent_name}-{workspace_id}"

        worktree_path = (
            self.worktree_root / branch_name
        )

        await _run_git(
            "worktree",
            "add",
            str(worktree_path),
            "-b",
            branch_name,
            cwd=self.repo_root,
        )

        return WorktreeInfo(
            workspace_id=workspace_id,
            branch_name=branch_name,
            path=worktree_path,
        )

    async def remove_worktree(
        self,
        info: WorktreeInfo,
    ) -> None:
        """
        Safely cleanup worktree + branch.
        """

        if info.path.exists():
            await _run_git(
                "worktree",
                "remove",
                str(info.path),
                "--force",
                cwd=self.repo_root,
            )

        await _run_git(
            "branch",
            "-D",
            info.branch_name,
            cwd=self.repo_root,
        )

    async def list_worktrees(
        self,
    ) -> list[WorktreeInfo]:
        """
        Return all managed worktrees.
        """

        output = await _run_git(
            "worktree",
            "list",
            "--porcelain",
            cwd=self.repo_root,
        )

        worktrees: list[WorktreeInfo] = []

        current: dict[str, str] = {}

        def flush_current() -> None:
            nonlocal current

            if not current:
                return

            path = Path(current["path"]).resolve()

            branch = current.get("branch", "")

            if (
                path.parent == self.worktree_root
                and branch
            ):
                parts = branch.rsplit("-", 1)

                workspace_id = (
                    parts[1]
                    if len(parts) == 2
                    else ""
                )

                worktrees.append(
                    WorktreeInfo(
                        workspace_id=workspace_id,
                        branch_name=branch,
                        path=path,
                    )
                )

            current = {}

        for line in output.splitlines():

            if line.startswith("worktree "):
                current = {
                    "path": line.split(" ", 1)[1]
                }

            elif line.startswith("branch "):
                current["branch"] = (
                    line.split(
                        "refs/heads/",
                        1,
                    )[-1]
                )

            elif line == "":
                flush_current()

        # Flush final block if no trailing newline
        flush_current()

        return worktrees