from __future__ import annotations
from dataclasses import dataclass, field
from backend.runtime.autonomous_patch import AutonomousPatchGenerator
from backend.runtime.diff_synthesizer import DiffSynthesizer
from backend.runtime.output_parser import OutputParser
from backend.runtime.patch_sandbox import PatchSandbox
from backend.runtime.patches import StructuredPatch
from backend.runtime.validation import PatchValidator


@dataclass(slots=True)
class PipelineResult:
    diff: str
    summary: str
    reasoning: str
    risk: str
    impacted_files: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class AutonomousExecutionPipeline:
    """
    Full autonomous patch pipeline.
    Stages:
    1. Generate patch via LLM cognition.
    2. Validate structured patch.
    3. Parse unified diff into structured output.
    4. Materialize files into sandbox.
    5. Synthesize git diff.
    6. Return PipelineResult.
    """

    def __init__(
        self,
        generator: AutonomousPatchGenerator,
        parser: OutputParser,
        validator: PatchValidator,
        sandbox: PatchSandbox,
        synthesizer: DiffSynthesizer,
    ) -> None:
        self.generator = generator
        self.parser = parser
        self.validator = validator
        self.sandbox = sandbox
        self.synthesizer = synthesizer

    async def run(
        self,
        *,
        task: str,
        repository_context: str,
        impacted_files: list[str],
        agent_id: str = "coder-agent",
    ) -> PipelineResult:
        # Stage 1: Generate
        patch = await self.generator.generate_patch(
            task=task,
            repository_context=repository_context,
            impacted_files=impacted_files,
            agent_id=agent_id,
        )


        # Stage 3: Parse
        parsed = self.parser.parse_patch_output(patch.unified_diff)

        # Stage 4: Materialize
        written = await self.sandbox.materialize_patch(parsed)
        if not written:
            raise ValueError("Sandbox materialization produced no files.")

        # Stage 5: Synthesize diff
        diff = await self.synthesizer.synthesize_diff()
        validated = self.validator.validate(
            patch.model_copy(
                update={
                    "unified_diff": diff
                }
            )
        )

        errors = getattr(
            validated,
            "validation_errors",
            [],
        )

        if errors:
            raise ValueError(
                f"Patch validation failed: "
                f"{errors}"
            )

        # Stage 6: Return
        return PipelineResult(
            diff=diff,
            summary=parsed.summary,
            reasoning=parsed.reasoning,
            risk=parsed.risk,
            impacted_files=[str(p) for p in written],
            metadata=patch.metadata if hasattr(patch, "metadata") else {},
        )

    async def run_safe(
        self,
        *,
        task: str,
        repository_context: str,
        impacted_files: list[str],
        agent_id: str = "coder-agent",
    ) -> PipelineResult | Exception:
        """
        Non-raising wrapper around run().
        Returns the result or the exception — caller decides how to handle.
        """
        try:
            return await self.run(
                task=task,
                repository_context=repository_context,
                impacted_files=impacted_files,
                agent_id=agent_id,
            )
        except Exception as exc:
            return exc