import pytest
from backend.llm.providers.openai_compatible import OpenAICompatibleClient
from backend.runtime.autonomous_patch import AutonomousPatchGenerator
from backend.runtime.live_cognition import LiveCognition, LiveCognitionResponse


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.persistent_runtime
async def test_live_patch_generation():
    client = OpenAICompatibleClient(
        base_url="http://localhost:8010",
        model="qwen-primary",
    )
    cognition = LiveCognition(primary_client=client)
    generator = AutonomousPatchGenerator(cognition)

    patch = await generator.generate_patch(
        task="Add Python type hints to the hello function.",
        repository_context="""
        File: hello.py
        def hello(name):
            return "hello " + name
        """,
        impacted_files=["hello.py"],
    )

    # Debug output
    print()
    print("========== GENERATED PATCH ==========")
    print(patch.unified_diff)
    print("=====================================")
    print()

    # Assertions
    assert isinstance(patch.unified_diff, str), "unified_diff must be a string"
    assert len(patch.unified_diff) > 0, "unified_diff must not be empty"
    assert "hello.py" in [
        target.path for target in patch.impacted_files
    ], "hello.py must appear in impacted_files"

    await cognition.close()


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.persistent_runtime
async def test_live_patch_generation_context_manager():
    """Same flow but using async context manager for clean teardown."""
    client = OpenAICompatibleClient(
        base_url="http://localhost:8010",
        model="qwen-primary",
    )

    async with LiveCognition(primary_client=client) as cognition:
        generator = AutonomousPatchGenerator(cognition)

        patch = await generator.generate_patch(
            task="Add a docstring to the hello function.",
            repository_context="""
            File: hello.py
            def hello(name):
                return "hello " + name
            """,
            impacted_files=["hello.py"],
        )

    assert isinstance(patch.unified_diff, str)
    assert len(patch.unified_diff) > 0
    assert "hello.py" in [t.path for t in patch.impacted_files]
