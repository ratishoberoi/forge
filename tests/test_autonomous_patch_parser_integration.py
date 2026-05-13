import pytest
from backend.runtime.autonomous_patch import AutonomousPatchGenerator
from backend.runtime.live_cognition import LiveCognitionResponse
from backend.runtime.output_parser import OutputParser
from backend.runtime.patches import PatchRisk


VALID_LLM_RESPONSE = """
Thinking out loud...
{
  "summary": "Add typing",
  "reasoning": "Improves clarity",
  "risk": "low",
  "files": {
    "hello.py": "def hello(name: str) -> str: return 'hello ' + name"
  }
}
"""


class FakeCognition:
    """Minimal fake that returns a hardcoded LiveCognitionResponse."""

    async def complete(self, **kwargs) -> LiveCognitionResponse:
        return LiveCognitionResponse(
            content=VALID_LLM_RESPONSE,
            model="primary_coder",
            prompt_tokens=10,
            completion_tokens=20,
            finish_reason="stop",
        )


class MalformedCognition:
    """Returns unparseable content — simulates bad LLM output."""

    async def complete(self, **kwargs) -> LiveCognitionResponse:
        return LiveCognitionResponse(
            content="Sorry, I cannot help with that.",
            model="primary_coder",
            prompt_tokens=5,
            completion_tokens=3,
            finish_reason="stop",
        )


class MissingFieldsCognition:
    """Returns JSON but missing required fields."""

    async def complete(self, **kwargs) -> LiveCognitionResponse:
        return LiveCognitionResponse(
            content='{ "summary": "only summary" }',
            model="primary_coder",
            prompt_tokens=5,
            completion_tokens=5,
            finish_reason="stop",
        )


# ── Happy path ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_patch_generator_uses_parser():
    generator = AutonomousPatchGenerator(
        cognition=FakeCognition(),
        parser=OutputParser(),
    )
    patch = await generator.generate_patch(
        task="Add typing",
        repository_context="hello.py",
        impacted_files=["hello.py"],
    )

    assert patch.title == "Add typing"
    assert patch.description == "Improves clarity"
    assert patch.risk == PatchRisk.LOW
    assert patch.risk.value == "low"
    assert patch.impacted_files[0].path == "hello.py"


@pytest.mark.asyncio
async def test_patch_generator_impacted_files_from_parsed_json():
    """impacted_files must come from parsed.files.keys(), not caller input."""
    generator = AutonomousPatchGenerator(
        cognition=FakeCognition(),
        parser=OutputParser(),
    )
    # Pass a different file — patch should still use what JSON says
    patch = await generator.generate_patch(
        task="Add typing",
        repository_context="hello.py",
        impacted_files=["unrelated.py"],
    )

    paths = [t.path for t in patch.impacted_files]
    assert "hello.py" in paths
    assert "unrelated.py" not in paths


@pytest.mark.asyncio
async def test_patch_metadata_preserved():
    generator = AutonomousPatchGenerator(
        cognition=FakeCognition(),
        parser=OutputParser(),
    )
    patch = await generator.generate_patch(
        task="Add typing",
        repository_context="hello.py",
        impacted_files=["hello.py"],
    )

    assert patch.metadata["model"] == "primary_coder"
    assert patch.metadata["prompt_tokens"] == 10
    assert patch.metadata["completion_tokens"] == 20
    assert patch.metadata["finish_reason"] == "stop"


# ── Failure paths ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_patch_generator_raises_on_malformed_output():
    generator = AutonomousPatchGenerator(
        cognition=MalformedCognition(),
        parser=OutputParser(),
    )
    with pytest.raises(ValueError, match="No JSON payload found"):
        await generator.generate_patch(
            task="Add typing",
            repository_context="hello.py",
            impacted_files=["hello.py"],
        )


@pytest.mark.asyncio
async def test_patch_generator_raises_on_missing_fields():
    generator = AutonomousPatchGenerator(
        cognition=MissingFieldsCognition(),
        parser=OutputParser(),
    )
    with pytest.raises(ValueError, match="Missing fields"):
        await generator.generate_patch(
            task="Add typing",
            repository_context="hello.py",
            impacted_files=["hello.py"],
        )