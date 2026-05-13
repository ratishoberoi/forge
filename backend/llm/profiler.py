from __future__ import annotations
from dataclasses import dataclass
from backend.llm.registry import ModelRecord


@dataclass(slots=True)
class ModelProfile:
    alias: str
    estimated_vram_gb: float
    context_window: int
    quantization: str | None
    reasoning: bool
    moe: bool
    architecture_focus: bool
    recommended_concurrency: int
    estimated_tokens_per_second: int


class ModelProfiler:
    """
    Hardware-aware model profiler.
    Initially uses static estimates.
    Future:
    - live benchmarking
    - real VRAM measurement
    - throughput profiling
    """

    def profile(self, record: ModelRecord) -> ModelProfile:
        vram_estimate = self._estimate_vram(record)
        return ModelProfile(
            alias=record.alias,
            estimated_vram_gb=vram_estimate,
            context_window=record.runtime.max_model_len or 4096,
            quantization=record.quantization.method,
            reasoning=record.reasoning,
            moe=record.moe,
            architecture_focus=record.architecture_focus,
            recommended_concurrency=self._estimate_concurrency(vram_estimate),
            estimated_tokens_per_second=self._estimate_tps(record),
        )

    def _estimate_vram(self, record: ModelRecord) -> float:
        name = record.model_name.lower()
        if "35b" in name:
            return 24.0
        if "32b" in name:
            return 22.0
        if "moe" in name:
            return 18.0
        return 12.0

    def _estimate_tps(self, record: ModelRecord) -> int:
        name = record.model_name.lower()
        if "35b" in name:
            return 28
        if "32b" in name:
            return 34
        if "moe" in name:
            return 48
        return 60

    def _estimate_concurrency(self, vram_gb: float) -> int:
        if vram_gb >= 24:
            return 1
        if vram_gb >= 20:
            return 2
        return 3