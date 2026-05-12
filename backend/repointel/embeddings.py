"""Local embedding pipeline with SQLite caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

from sentence_transformers import SentenceTransformer

from backend.config.settings import Settings
from backend.config.settings import get_settings
from backend.core.errors import ConfigurationError
from backend.llm.registry import ModelRegistry
from backend.repointel.models import CodeChunk


class EmbeddingCache:
    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, cache_key: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put(self, cache_key: str, vector: list[float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings(cache_key, vector) VALUES(?, ?)",
            (cache_key, json.dumps(vector)),
        )
        self._conn.commit()


class EmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._registry = ModelRegistry(self._settings)
        self._cache = EmbeddingCache(self._settings.embedding_cache_path)
        self._model: SentenceTransformer | None = None

    async def initialize(self) -> None:
        await self._load_model()

    async def validate_startup(self) -> dict[str, object]:
        try:
            return await self.healthcheck()
        except Exception as exc:
            raise ConfigurationError(
                "EmbeddingService startup validation failed. "
                "Ensure the configured embedding model is available locally and readable."
            ) from exc

    async def embed_text(self, text: str, *, cache_key: str | None = None) -> list[float]:
        embeddings = await self.embed_texts({cache_key or "text": text})
        return embeddings[cache_key or "text"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {f"text-{index}": text for index, text in enumerate(texts)}
        embedded = await self.embed_texts(payload)
        return [embedded[key] for key in payload]

    async def embed_chunks(self, chunks: list[CodeChunk]) -> dict[str, list[float]]:
        if not chunks:
            return {}
        texts = {chunk.id: chunk.content for chunk in chunks}
        return await self.embed_texts(texts)

    async def embed_texts(self, texts: dict[str, str]) -> dict[str, list[float]]:
        cached: dict[str, list[float]] = {}
        uncached_ids: list[str] = []
        uncached_texts: list[str] = []
        for item_id, text in texts.items():
            cache_key = self._cache_key(text)
            vector = self._cache.get(cache_key)
            if vector is not None:
                cached[item_id] = vector
            else:
                uncached_ids.append(item_id)
                uncached_texts.append(text)

        if uncached_ids:
            model = await self._load_model()
            embeddings = await asyncio.to_thread(
                model.encode,
                uncached_texts,
                batch_size=self._settings.embedding_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            for item_id, text, vector in zip(uncached_ids, uncached_texts, embeddings, strict=True):
                dense = [float(value) for value in vector.tolist()]
                cached[item_id] = dense
                self._cache.put(self._cache_key(text), dense)
        return cached

    async def _load_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        record = self._registry.embedding_model()
        try:
            self._model = await asyncio.to_thread(
                SentenceTransformer,
                record.model_name,
                trust_remote_code=record.trust_remote_code,
                local_files_only=True,
            )
        except Exception as exc:
            raise ConfigurationError(
                f"Failed to initialize embedding model '{record.model_name}'. "
                "Forge requires the embedding model to be available locally."
            ) from exc
        return self._model

    async def healthcheck(self) -> dict[str, object]:
        model = await self._load_model()
        vector = await self.embed_text("forge embedding healthcheck", cache_key="healthcheck")
        return {
            "model_name": self._registry.embedding_model().model_name,
            "embedding_dimension": len(vector),
            "device": str(getattr(model, "device", "unknown")),
        }

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


LocalEmbeddingPipeline = EmbeddingService

__all__ = ["EmbeddingCache", "EmbeddingService", "LocalEmbeddingPipeline"]
