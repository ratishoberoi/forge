import pytest
from backend.llm.router import ModelConfig, ModelRole, ModelRouter
from backend.config.settings import Settings


class FakeLLMService:
    async def generate_text(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> str:
        return f"model={model} temp={temperature}"


@pytest.fixture()
def router() -> ModelRouter:
    router = ModelRouter(llm_service=FakeLLMService(), settings=Settings())
    router.register_model(ModelConfig(name="qwen-primary", role=ModelRole.PRIMARY_CODER))
    router.register_model(ModelConfig(name="glm-judge", role=ModelRole.JUDGE, temperature=0.0))
    return router


def test_model_registration(router: ModelRouter):
    assert router.get_model(ModelRole.PRIMARY_CODER).name == "qwen-primary"


def test_missing_model():
    router = ModelRouter(llm_service=FakeLLMService(), settings=Settings())
    router.models.pop(ModelRole.RETRY_ENGINE)
    with pytest.raises(RuntimeError):
        router.get_model(ModelRole.RETRY_ENGINE)


@pytest.mark.asyncio
async def test_generation_dispatch(router: ModelRouter):
    result = await router.generate(role=ModelRole.PRIMARY_CODER, prompt="hello")
    assert "qwen-primary" in result


@pytest.mark.asyncio
async def test_judge_temperature(router: ModelRouter):
    result = await router.generate(role=ModelRole.JUDGE, prompt="judge")
    assert "temp=0.0" in result


def test_overwrite_registration(router: ModelRouter):
    """Re-registering a role must replace the old config."""
    router.register_model(ModelConfig(name="new-model", role=ModelRole.PRIMARY_CODER))
    assert router.get_model(ModelRole.PRIMARY_CODER).name == "new-model"


def test_register_registry_models():
    router = ModelRouter(llm_service=FakeLLMService(), settings=Settings())
    assert router.get_model(ModelRole.PRIMARY_CODER).name == "Qwen3.5-Coder-35B-A3B"
    assert router.get_model(ModelRole.JUDGE).name == "GLM-Reasoning-Judge"