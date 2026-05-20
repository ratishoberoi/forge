import pytest
from backend.runtime.autonomous_patch import AutonomousPatchGenerator
from backend.runtime.cognition import CognitionAdapter, CognitionResponse
from backend.llm.router import ModelRole


class FakeCognitionAdapter:
    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_role: ModelRole = ModelRole.PRIMARY_CODER,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        agent_id: str | None = None,
    ) -> CognitionResponse:
        return CognitionResponse(
            content="diff --git a/hello.py b/hello.py\n+def hello(name: str) -> str:",
            model=model_role.value,
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_patch_generation():
    generator = AutonomousPatchGenerator(cognition=FakeCognitionAdapter())
    patch = await generator.generate_patch(
        task="Add type hints to a Python hello world function.",
        repository_context="File: hello.py\ndef hello(name):\n    return 'hello ' + name",
        impacted_files=["hello.py"],
    )
    assert isinstance(patch.unified_diff, str)
    assert len(patch.unified_diff) > 0
    assert "hello.py" in [t.path for t in patch.impacted_files]


@pytest.mark.asyncio
async def test_patch_metadata_populated():
    generator = AutonomousPatchGenerator(cognition=FakeCognitionAdapter())
    patch = await generator.generate_patch(
        task="test task",
        repository_context="ctx",
        impacted_files=["app.py"],
        agent_id="test-agent",
    )
    assert patch.metadata["agent_id"] == "test-agent"
    assert patch.metadata["model"] == ModelRole.PRIMARY_CODER.value
    assert patch.metadata["prompt_tokens"] == 100
    assert patch.metadata["completion_tokens"] == 50


@pytest.mark.asyncio
async def test_patch_is_validated():
    generator = AutonomousPatchGenerator(cognition=FakeCognitionAdapter())
    patch = await generator.generate_patch(
        task="test",
        repository_context="ctx",
        impacted_files=["hello.py"],
    )
    assert patch.is_valid()


@pytest.mark.asyncio
async def test_patch_title_matches_task():
    generator = AutonomousPatchGenerator(cognition=FakeCognitionAdapter())
    patch = await generator.generate_patch(
        task="unique-task-abc123",
        repository_context="ctx",
        impacted_files=["f.py"],
    )
    assert patch.title == "unique-task-abc123"


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.persistent_runtime
async def test_real_patch_generation():
    """Live test — requires running LLM server. Skip in CI."""
    from backend.config.settings import Settings
    from backend.llm.engine import LLMEngineManager
    from backend.llm.service import ChatCompletionService

    settings = Settings()
    engine_manager = LLMEngineManager(settings)
    service = ChatCompletionService(settings=settings, engine_manager=engine_manager)
    cognition = CognitionAdapter(service)
    generator = AutonomousPatchGenerator(cognition)

    patch = await generator.generate_patch(
        task="Add type hints to a Python hello world function.",
        repository_context="File: hello.py\ndef hello(name):\n    return 'hello ' + name",
        impacted_files=["hello.py"],
    )
    assert isinstance(patch.unified_diff, str)
    assert len(patch.unified_diff) > 0
    assert "hello.py" in [t.path for t in patch.impacted_files]
