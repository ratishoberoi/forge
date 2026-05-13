import pytest
from backend.config.settings import Settings
from backend.llm.registry import ModelRegistry
from backend.llm.runtime_manager import RuntimeManager


@pytest.fixture()
def runtime() -> RuntimeManager:
    return RuntimeManager(registry=ModelRegistry(settings=Settings()))


@pytest.mark.asyncio
async def test_load_model(runtime: RuntimeManager):
    loaded = await runtime.load_model("qwen-primary")
    assert loaded.record.alias == "qwen-primary"
    assert runtime.is_loaded("qwen-primary")


@pytest.mark.asyncio
async def test_unload_model(runtime: RuntimeManager):
    await runtime.load_model("qwen-primary")
    await runtime.unload_model("qwen-primary")
    assert not runtime.is_loaded("qwen-primary")
    assert runtime.active_model is None


@pytest.mark.asyncio
async def test_swap_model(runtime: RuntimeManager):
    await runtime.load_model("qwen-primary")
    await runtime.swap_model("glm-judge")
    assert runtime.active_model == "glm-judge"
    assert not runtime.is_loaded("qwen-primary")
    assert runtime.is_loaded("glm-judge")


@pytest.mark.asyncio
async def test_double_load(runtime: RuntimeManager):
    first = await runtime.load_model("qwen-primary")
    second = await runtime.load_model("qwen-primary")
    assert first is second


@pytest.mark.asyncio
async def test_loaded_aliases(runtime: RuntimeManager):
    await runtime.load_model("qwen-primary")
    await runtime.load_model("glm-judge")
    aliases = runtime.loaded_aliases()
    assert "qwen-primary" in aliases
    assert "glm-judge" in aliases


@pytest.mark.asyncio
async def test_unload_nonexistent_is_safe(runtime: RuntimeManager):
    """Unloading a model that was never loaded must not raise."""
    await runtime.unload_model("ghost-model")


@pytest.mark.asyncio
async def test_active_model_after_load(runtime: RuntimeManager):
    await runtime.load_model("qwen-primary")
    assert runtime.active_model == "qwen-primary"