from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from backend.llm.service import ChatCompletionService
from backend.config.settings import Settings
from backend.llm.registry import ModelRegistry


class ModelRole(StrEnum):
    PRIMARY_CODER = "primary_coder"
    REPO_SYNTHESIZER = "repo_synthesizer"
    RETRY_ENGINE = "retry_engine"
    ARCHITECTURE_CODER = "architecture_coder"
    JUDGE = "judge"


@dataclass(slots=True)
class ModelConfig:
    name: str
    role: ModelRole
    temperature: float = 0.1
    max_tokens: int = 4096


_ROLE_MAPPING: dict[str, ModelRole] = {
    "primary_coder": ModelRole.PRIMARY_CODER,
    "repo_synthesizer": ModelRole.REPO_SYNTHESIZER,
    "retry_engine": ModelRole.RETRY_ENGINE,
    "architecture_coder": ModelRole.ARCHITECTURE_CODER,
    "judge": ModelRole.JUDGE,
}


class ModelRouter:
    """
    Central model orchestration layer.
    Responsibilities:
    - model role routing
    - generation dispatch
    - unified model API
    """

    def __init__(
        self,
        llm_service: ChatCompletionService,
        settings: Settings,
    ) -> None:
        self.llm_service = llm_service
        self.registry = ModelRegistry(settings=settings)
        self.models: dict[ModelRole, ModelConfig] = {}
        self.register_registry_models()

    def register_registry_models(self) -> None:
        """Register all canonical models from the centralized registry."""
        for record in self.registry.list_models():
            if record.role == "embedding":
                continue
            self.register_model(ModelConfig(
                name=record.model_name,
                role=_ROLE_MAPPING[record.role],
                temperature=0.1,
                max_tokens=record.runtime.max_model_len or 4096,
            ))

    def register_model(self, config: ModelConfig) -> None:
        self.models[config.role] = config

    def get_model(self, role: ModelRole) -> ModelConfig:
        if role not in self.models:
            raise RuntimeError(f"No model registered for role: {role}")
        return self.models[role]

    async def generate(self, role: ModelRole, prompt: str) -> str:
        model = self.get_model(role)
        return await self.llm_service.generate_text(
            prompt=prompt,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            model=model.name,
        )