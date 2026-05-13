from __future__ import annotations
from backend.llm.router import ModelRole
from backend.runtime.live_cognition import LiveCognition
from backend.runtime.output_parser import OutputParser
from backend.runtime.patches import PatchRisk, PatchTarget, StructuredPatch
from backend.runtime.prompting import build_coder_system_prompt, build_patch_generation_prompt
from backend.runtime.validation import PatchValidator


class AutonomousPatchGenerator:
    """
    Real cognition-backed patch generator.
    """

    def __init__(
        self,
        cognition: LiveCognition,
        parser: OutputParser | None = None,
    ) -> None:
        self.cognition = cognition
        self.validator = PatchValidator()
        self.parser = parser or OutputParser()

    async def generate_patch(
        self,
        *,
        task: str,
        repository_context: str,
        impacted_files: list[str],
        agent_id: str = "coder-agent",
    ) -> StructuredPatch:
        response = await self.cognition.complete(
            system_prompt=build_coder_system_prompt(),
            user_prompt=build_patch_generation_prompt(
                task=task,
                repository_context=repository_context,
            ),
            model_role=ModelRole.PRIMARY_CODER,
            max_tokens=2048,
            agent_id=agent_id,
        )

        parsed = self.parser.parse_patch_output(response.content)

        patch = StructuredPatch(
            title=parsed.summary,
            description=parsed.reasoning,
            unified_diff=response.content,
            impacted_files=[
                PatchTarget(path=path)
                for path in parsed.files.keys()
            ],
            risk=PatchRisk(parsed.risk),
            metadata={
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "finish_reason": response.finish_reason,
                "agent_id": agent_id,
            },
        )

        return self.validator.validate(patch)