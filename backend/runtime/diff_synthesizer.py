from __future__ import annotations
from backend.runtime.gitops import GitOperations
from backend.runtime.output_parser import ParsedPatchOutput
from backend.runtime.patch_sandbox import PatchSandbox


class DiffSynthesizer:
    """
    Generates git diffs from sandboxed filesystem mutations.
    Responsibilities:
    - materialize parsed patch into sandbox
    - trigger git diff generation
    - validate diff is non-empty
    - return raw unified diff string
    """

    def __init__(
        self,
        sandbox: PatchSandbox,
        gitops: GitOperations,
    ) -> None:
        self.sandbox = sandbox
        self.gitops = gitops

    async def synthesize_diff(self) -> str:
        """Get diff for whatever is currently staged in the sandbox."""
        diff = await self.gitops.get_diff()
        if not diff.strip():
            raise ValueError("No git diff generated.")
        return diff

    async def materialize_and_diff(self, parsed: ParsedPatchOutput) -> str:
        """
        Full pipeline:
        1. Materialize parsed files into sandbox.
        2. Generate and return the resulting git diff.
        """
        written = await self.sandbox.materialize_patch(parsed)
        if not written:
            raise ValueError("No files were materialized into sandbox.")

        diff = await self.gitops.get_diff()
        if not diff.strip():
            raise ValueError(
                f"Files were written ({[str(p) for p in written]}) "
                "but git diff returned empty — check staging."
            )
        return diff

    async def materialize_stage_and_diff(self, parsed: ParsedPatchOutput) -> str:
        """
        Full pipeline with explicit git staging:
        1. Materialize parsed files into sandbox.
        2. Stage all written files via gitops.
        3. Generate and return the resulting git diff.
        """
        written = await self.sandbox.materialize_patch(parsed)
        if not written:
            raise ValueError("No files were materialized into sandbox.")

        await self.gitops.stage_files([str(p) for p in written])

        diff = await self.gitops.get_diff()
        if not diff.strip():
            raise ValueError(
                f"Files staged ({[str(p) for p in written]}) "
                "but git diff returned empty."
            )
        return diff