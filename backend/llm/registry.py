"""Centralized local model registry."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.config.settings import Settings
from backend.core.errors import InvalidRequestError


class QuantizationConfig(BaseModel):
    method: str | None = None
    bits: int | None = Field(default=None, ge=2, le=16)


class RuntimeModelMetadata(BaseModel):
    max_model_len: int | None = None
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.92, gt=0.0, le=1.0)
    supports_streaming: bool = True


class ModelRecord(BaseModel):
    id: str
    alias: str
    role: Literal[
        "primary_coder",
        "repo_synthesizer",
        "retry_engine",
        "architecture_coder",
        "judge",
        "embedding",
    ]
    engine: Literal["vllm", "sentence_transformers"]
    provider: Literal["local"] = "local"
    model_name: str
    trust_remote_code: bool = False
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    runtime: RuntimeModelMetadata = Field(default_factory=RuntimeModelMetadata)
    reasoning: bool = False
    moe: bool = False
    architecture_focus: bool = False


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._records = {
            "qwen-primary": ModelRecord(
                id="Qwen3.5-Coder-35B-A3B",
                alias="qwen-primary",
                role="primary_coder",
                engine="vllm",
                model_name="Qwen3.5-Coder-35B-A3B",
                quantization=QuantizationConfig(method="AWQ"),
                runtime=RuntimeModelMetadata(max_model_len=32768),
            ),
            "deepseek-synth": ModelRecord(
                id="DeepSeek-V4-class-32B",
                alias="deepseek-synth",
                role="repo_synthesizer",
                engine="vllm",
                model_name="DeepSeek-V4-class-32B",
                quantization=QuantizationConfig(method="GPTQ"),
                runtime=RuntimeModelMetadata(max_model_len=32768),
            ),
            "qwen-retry": ModelRecord(
                id="Qwen3-A3B-MoE",
                alias="qwen-retry",
                role="retry_engine",
                engine="vllm",
                model_name="Qwen3-A3B-MoE",
                quantization=QuantizationConfig(method="AWQ"),
                runtime=RuntimeModelMetadata(max_model_len=16384),
                moe=True,
            ),
            "glm-architect": ModelRecord(
                id="GLM-4.7-class",
                alias="glm-architect",
                role="architecture_coder",
                engine="vllm",
                model_name="GLM-4.7-class",
                quantization=QuantizationConfig(method="GPTQ"),
                runtime=RuntimeModelMetadata(max_model_len=32768),
                architecture_focus=True,
            ),
            "glm-judge": ModelRecord(
                id="GLM-Reasoning-Judge",
                alias="glm-judge",
                role="judge",
                engine="vllm",
                model_name="GLM-Reasoning-Judge",
                quantization=QuantizationConfig(method="GPTQ"),
                runtime=RuntimeModelMetadata(max_model_len=32768),
                reasoning=True,
            ),
            settings.embedding_model_alias: ModelRecord(
                id=settings.embedding_model_name,
                alias=settings.embedding_model_alias,
                role="embedding",
                engine="sentence_transformers",
                model_name=settings.embedding_model_name,
            ),
        }

    def generation_model(self) -> ModelRecord:
        return self._records[self._settings.model_alias]

    def embedding_model(self) -> ModelRecord:
        return self._records[self._settings.embedding_model_alias]

    def resolve_generation_model(self, requested_model: str | None) -> ModelRecord:
        generation = self.generation_model()
        supported = {generation.alias, generation.id}
        if requested_model is None or requested_model in supported:
            return generation
        raise InvalidRequestError(
            f"Unsupported model '{requested_model}'.",
            details={"supported_models": sorted(supported)},
        )

    def list_models(self) -> list[ModelRecord]:
        return list(self._records.values())