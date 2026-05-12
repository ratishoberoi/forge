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
    role: Literal["generation", "embedding"]
    engine: Literal["vllm", "sentence_transformers"]
    provider: Literal["local"] = "local"
    model_name: str
    trust_remote_code: bool = False
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    runtime: RuntimeModelMetadata = Field(default_factory=RuntimeModelMetadata)


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._records = {
            settings.model_alias: ModelRecord(
                id=settings.model_name,
                alias=settings.model_alias,
                role="generation",
                engine="vllm",
                model_name=settings.model_name,
                trust_remote_code=settings.model_trust_remote_code,
                quantization=QuantizationConfig(method=settings.model_quantization),
                runtime=RuntimeModelMetadata(
                    max_model_len=settings.model_max_model_len,
                    tensor_parallel_size=settings.model_tensor_parallel_size,
                    gpu_memory_utilization=settings.model_gpu_memory_utilization,
                    supports_streaming=True,
                ),
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
