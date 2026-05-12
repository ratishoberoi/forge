"""vLLM engine lifecycle management."""

from __future__ import annotations

import asyncio
import logging

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import AsyncEngineArgs, AsyncLLMEngine

from backend.config.settings import Settings
from backend.llm.registry import ModelRegistry

logger = logging.getLogger(__name__)


class LLMEngineManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._registry = ModelRegistry(settings)
        self._engine: AsyncLLMEngine | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._init_lock = asyncio.Lock()

    @property
    def engine(self) -> AsyncLLMEngine | None:
        return self._engine

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        return self._tokenizer

    async def initialize(self) -> None:
        if self._engine and self._tokenizer:
            return

        async with self._init_lock:
            if self._engine and self._tokenizer:
                return

            generation_model = self._registry.generation_model()
            logger.info("Initializing tokenizer for model %s", generation_model.model_name)
            self._tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                generation_model.model_name,
                trust_remote_code=generation_model.trust_remote_code,
            )

            engine_args = AsyncEngineArgs(
                model=generation_model.model_name,
                served_model_name=generation_model.alias,
                trust_remote_code=generation_model.trust_remote_code,
                dtype=self._settings.model_dtype,
                max_model_len=self._settings.model_max_model_len,
                gpu_memory_utilization=self._settings.model_gpu_memory_utilization,
                tensor_parallel_size=self._settings.model_tensor_parallel_size,
                max_num_seqs=self._settings.model_max_num_seqs,
                enable_prefix_caching=self._settings.model_enable_prefix_caching,
                download_dir=self._settings.model_download_dir,
                enforce_eager=self._settings.model_enforce_eager,
                quantization=self._settings.model_quantization,
            )

            logger.info("Starting vLLM engine for model %s", generation_model.model_name)
            self._engine = await asyncio.to_thread(AsyncLLMEngine.from_engine_args, engine_args)

    async def shutdown(self) -> None:
        if self._engine is None:
            return

        logger.info("Shutting down vLLM engine")
        await asyncio.to_thread(self._engine.shutdown)
        self._engine = None
        self._tokenizer = None
