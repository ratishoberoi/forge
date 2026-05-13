import pytest

from backend.runtime.diff_synthesizer import (
    DiffSynthesizer,
)
from backend.runtime.output_parser import (
    OutputParser,
)
from backend.runtime.patch_sandbox import (
    PatchSandbox,
)
from backend.runtime.pipeline import (
    AutonomousExecutionPipeline,
)
from backend.runtime.validation import (
    PatchValidator,
)


class FakeGenerator:

    async def generate_patch(
        self,
        **kwargs,
    ):

        from backend.runtime.patches import (
            PatchRisk,
            PatchTarget,
            StructuredPatch,
        )

        return StructuredPatch(
            title="Add typing",
            description="Safe typing",
            unified_diff="""
            {
              "summary": "Add typing",
              "reasoning": "Improves safety",
              "risk": "low",
              "files": {
                "hello.py":
                "def hello(name: str) -> str:\\n    return 'hello ' + name\\n"
              }
            }
            """,
            impacted_files=[
                PatchTarget(
                    path="hello.py"
                )
            ],
            risk=PatchRisk.LOW,
        )


class FakeGitOps:

    async def get_diff(
        self,
    ):

        return (
            "diff --git a/hello.py "
            "b/hello.py"
        )


@pytest.mark.asyncio
async def test_pipeline_full_flow(
    tmp_path,
):

    sandbox = PatchSandbox(
        str(tmp_path)
    )

    pipeline = (
        AutonomousExecutionPipeline(
            generator=FakeGenerator(),
            parser=OutputParser(),
            validator=PatchValidator(),
            sandbox=sandbox,
            synthesizer=(
                DiffSynthesizer(
                    sandbox=sandbox,
                    gitops=FakeGitOps(),
                )
            ),
        )
    )

    result = await (
        pipeline.run(
            task="Add typing",
            repository_context="hello.py",
            impacted_files=[
                "hello.py"
            ],
        )
    )

    assert (
        "diff --git"
        in result.diff
    )

    assert (
        result.summary
        == "Add typing"
    )

    assert (
        result.risk
        == "low"
    )