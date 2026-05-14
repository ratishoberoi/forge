"""Centralized cognition runtime lifecycle state management."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum

from backend.llm.router import ModelRole


class RuntimeState(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOADING = "loading"
    FAILED = "failed"
    UNHEALTHY = "unhealthy"


@dataclass(slots=True)
class RuntimeEndpoint:
    base_url: str
    model_name: str
    api_path: str = "/v1/chat/completions"


@dataclass(slots=True)
class LoadedRuntime:
    runtime_id: str
    role: ModelRole
    endpoint: RuntimeEndpoint
    state: RuntimeState = RuntimeState.INACTIVE
    owner: str | None = None
    activation_timestamp: float | None = None
    process_metadata: dict[str, object] = field(default_factory=dict)
    quantization_metadata: dict[str, object] = field(default_factory=dict)
    memory_metadata: dict[str, object] = field(default_factory=dict)
    lightweight: bool = False

    @property
    def health(self) -> RuntimeState:
        return self.state


class CognitionLifecycleManager:
    """Async-safe registry and transition manager for cognition runtimes."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._runtimes: dict[str, LoadedRuntime] = {}

    async def register_runtime(
        self,
        *,
        runtime_id: str,
        role: ModelRole,
        endpoint: RuntimeEndpoint,
        owner: str | None = None,
        process_metadata: dict[str, object] | None = None,
        quantization_metadata: dict[str, object] | None = None,
        memory_metadata: dict[str, object] | None = None,
        lightweight: bool = False,
        state: RuntimeState = RuntimeState.INACTIVE,
    ) -> LoadedRuntime:
        async with self._lock:
            runtime = LoadedRuntime(
                runtime_id=runtime_id,
                role=role,
                endpoint=endpoint,
                state=state,
                owner=owner,
                process_metadata=dict(process_metadata or {}),
                quantization_metadata=dict(quantization_metadata or {}),
                memory_metadata=dict(memory_metadata or {}),
                lightweight=lightweight,
            )
            self._runtimes[runtime_id] = runtime
            return runtime

    async def activate_runtime(self, runtime_id: str, *, owner: str | None = None) -> LoadedRuntime:
        async with self._lock:
            runtime = self._require(runtime_id)
            if runtime.state is RuntimeState.ACTIVE:
                if owner is not None:
                    runtime.owner = owner
                if runtime.activation_timestamp is None:
                    runtime.activation_timestamp = time.time()
                return runtime

            if not runtime.lightweight:
                for existing in self._runtimes.values():
                    if existing.runtime_id == runtime_id:
                        continue
                    if existing.state is RuntimeState.ACTIVE and not existing.lightweight:
                        existing.state = RuntimeState.INACTIVE
                        existing.owner = None

            runtime.state = RuntimeState.ACTIVE
            runtime.owner = owner
            runtime.activation_timestamp = time.time()
            return runtime

    async def deactivate_runtime(self, runtime_id: str) -> LoadedRuntime:
        async with self._lock:
            runtime = self._require(runtime_id)
            if runtime.state is RuntimeState.INACTIVE:
                runtime.owner = None
                return runtime
            runtime.state = RuntimeState.INACTIVE
            runtime.owner = None
            return runtime

    async def swap_runtime(
        self,
        current_runtime_id: str,
        next_runtime_id: str,
        *,
        owner: str | None = None,
    ) -> LoadedRuntime:
        async with self._lock:
            current = self._require(current_runtime_id)
            next_runtime = self._require(next_runtime_id)
            if current.runtime_id != next_runtime.runtime_id:
                current.state = RuntimeState.INACTIVE
                current.owner = None
            next_runtime.state = RuntimeState.ACTIVE
            next_runtime.owner = owner
            if next_runtime.activation_timestamp is None or next_runtime.runtime_id != current_runtime_id:
                next_runtime.activation_timestamp = time.time()
            if not next_runtime.lightweight:
                for existing in self._runtimes.values():
                    if existing.runtime_id == next_runtime.runtime_id:
                        continue
                    if existing.state is RuntimeState.ACTIVE and not existing.lightweight:
                        existing.state = RuntimeState.INACTIVE
                        existing.owner = None
            return next_runtime

    async def active_runtime(self) -> LoadedRuntime | None:
        async with self._lock:
            return self._select_active_runtime()

    async def runtime_health(self, runtime_id: str) -> RuntimeState:
        async with self._lock:
            return self._require(runtime_id).health

    async def runtime_for_role(self, role: ModelRole) -> LoadedRuntime | None:
        async with self._lock:
            return self._select_runtime_for_role(role)

    async def inference_endpoint(
        self,
        *,
        role: ModelRole | None = None,
        runtime_id: str | None = None,
    ) -> RuntimeEndpoint | None:
        async with self._lock:
            if runtime_id is not None:
                return self._require(runtime_id).endpoint
            if role is not None:
                runtime = self._select_runtime_for_role(role)
                return runtime.endpoint if runtime is not None else None
            active = self._select_active_runtime()
            return active.endpoint if active is not None else None

    def _require(self, runtime_id: str) -> LoadedRuntime:
        try:
            return self._runtimes[runtime_id]
        except KeyError as exc:
            raise KeyError(f"Unknown runtime_id '{runtime_id}'.") from exc

    def _select_active_runtime(self) -> LoadedRuntime | None:
        active = [
            runtime
            for runtime in self._runtimes.values()
            if runtime.state is RuntimeState.ACTIVE and not runtime.lightweight
        ]
        if active:
            active.sort(
                key=lambda runtime: (runtime.activation_timestamp or 0.0, runtime.runtime_id),
                reverse=True,
            )
            return active[0]
        lightweight = [runtime for runtime in self._runtimes.values() if runtime.state is RuntimeState.ACTIVE]
        if not lightweight:
            return None
        lightweight.sort(
            key=lambda runtime: (runtime.activation_timestamp or 0.0, runtime.runtime_id),
            reverse=True,
        )
        return lightweight[0]

    def _select_runtime_for_role(self, role: ModelRole) -> LoadedRuntime | None:
        matching = [runtime for runtime in self._runtimes.values() if runtime.role is role]
        if not matching:
            return None
        active = [runtime for runtime in matching if runtime.state is RuntimeState.ACTIVE]
        if active:
            active.sort(
                key=lambda runtime: (runtime.activation_timestamp or 0.0, runtime.runtime_id),
                reverse=True,
            )
            return active[0]
        matching.sort(key=lambda runtime: runtime.runtime_id)
        return matching[0]
