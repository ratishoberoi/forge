from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from backend.llm.registry import ModelRecord, ModelRegistry


@dataclass(slots=True)
class LoadedModel:
    record: ModelRecord
    loaded_at: float


class RuntimeManager:
    """
    Dynamic local model runtime manager.
    Responsibilities:
    - load models
    - unload models
    - track active cognition
    - future GPU-aware orchestration
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.loaded_models: dict[str, LoadedModel] = {}
        self.active_model: str | None = None
        self._lock = asyncio.Lock()

    async def load_model(self, alias: str) -> LoadedModel:
        async with self._lock:
            if alias in self.loaded_models:
                return self.loaded_models[alias]

            record = self.registry._records[alias]

            # FUTURE: actual vLLM model loading

            loaded = LoadedModel(record=record, loaded_at=time.time())
            self.loaded_models[alias] = loaded
            self.active_model = alias
            return loaded

    async def unload_model(self, alias: str) -> None:
        async with self._lock:
            if alias not in self.loaded_models:
                return

            # FUTURE: actual GPU unload

            del self.loaded_models[alias]
            if self.active_model == alias:
                self.active_model = None

    async def swap_model(self, alias: str) -> LoadedModel:
        async with self._lock:
            if self.active_model and self.active_model != alias:
                # inline unload — no nested lock
                del self.loaded_models[self.active_model]
                self.active_model = None

            if alias in self.loaded_models:
                return self.loaded_models[alias]

            record = self.registry._records[alias]
            loaded = LoadedModel(record=record, loaded_at=time.time())
            self.loaded_models[alias] = loaded
            self.active_model = alias
            return loaded

    def is_loaded(self, alias: str) -> bool:
        return alias in self.loaded_models

    def loaded_aliases(self) -> list[str]:
        return list(self.loaded_models.keys())