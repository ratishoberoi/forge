import pytest
from backend.config.settings import Settings
from backend.llm.profiler import ModelProfiler
from backend.llm.registry import ModelRegistry


@pytest.fixture()
def ctx():
    return ModelRegistry(settings=Settings()), ModelProfiler()


def test_profile_primary(ctx):
    registry, profiler = ctx
    profile = profiler.profile(registry._records["qwen-primary"])
    assert profile.alias == "qwen-primary"
    assert profile.estimated_vram_gb >= 20
    assert profile.context_window == 32768
    assert profile.quantization == "AWQ"


def test_profile_moe(ctx):
    registry, profiler = ctx
    profile = profiler.profile(registry._records["qwen-retry"])
    assert profile.moe is True
    assert profile.recommended_concurrency >= 2
    assert profile.estimated_tokens_per_second >= 40


def test_profile_reasoning(ctx):
    registry, profiler = ctx
    profile = profiler.profile(registry._records["glm-judge"])
    assert profile.reasoning is True
    assert profile.estimated_tokens_per_second > 0


def test_profile_architecture_focus(ctx):
    registry, profiler = ctx
    profile = profiler.profile(registry._records["glm-architect"])
    assert profile.architecture_focus is True


def test_profile_concurrency_large_model(ctx):
    registry, profiler = ctx
    profile = profiler.profile(registry._records["qwen-primary"])
    assert profile.recommended_concurrency == 1