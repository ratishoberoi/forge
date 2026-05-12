"""Shared context and memory snapshot layer."""

from __future__ import annotations

import asyncio
import time
from pydantic import BaseModel, Field

from backend.config.settings import Settings, get_settings
from backend.repointel.models import ContextPackage
from backend.repointel.service import RepositoryIntelligenceEngine


class MemorySnapshot(BaseModel):
    key: str
    value: str
    created_at: float = Field(default_factory=time.time)


class SharedContextEnvelope(BaseModel):
    task_id: str
    summary: str
    repository_context: ContextPackage | None = None
    snapshots: list[MemorySnapshot] = Field(default_factory=list)
    estimated_tokens: int = 0


class SharedContextStore:
    def __init__(
        self,
        settings: Settings | None = None,
        repo_intelligence: RepositoryIntelligenceEngine | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._repo_intelligence = repo_intelligence
        self._snapshots: dict[str, list[MemorySnapshot]] = {}
        self._contexts: dict[str, SharedContextEnvelope] = {}
        self._lock = asyncio.Lock()

    async def put_snapshot(self, task_id: str, key: str, value: str) -> MemorySnapshot:
        snapshot = MemorySnapshot(key=key, value=value)
        async with self._lock:
            self._snapshots.setdefault(task_id, []).append(snapshot)
        return snapshot

    async def get_snapshots(self, task_id: str) -> list[MemorySnapshot]:
        async with self._lock:
            return list(self._snapshots.get(task_id, []))

    async def build_context(self, task_id: str, query: str, *, summary: str = "") -> SharedContextEnvelope:
        snapshots = await self.get_snapshots(task_id)
        repo_context = None
        if self._repo_intelligence is not None:
            repo_context = await self._repo_intelligence.build_context(query)
        estimated_tokens = self._estimate_tokens(summary)
        if repo_context is not None:
            estimated_tokens += self._estimate_tokens(" ".join(repo_context.related_files))
        estimated_tokens += sum(self._estimate_tokens(snapshot.value) for snapshot in snapshots)
        envelope = SharedContextEnvelope(
            task_id=task_id,
            summary=summary or query,
            repository_context=repo_context,
            snapshots=snapshots,
            estimated_tokens=min(estimated_tokens, self._settings.runtime_context_token_budget),
        )
        async with self._lock:
            self._contexts[task_id] = envelope
        return envelope

    async def get_context(self, task_id: str) -> SharedContextEnvelope | None:
        async with self._lock:
            return self._contexts.get(task_id)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4) if text else 0
