"""Local Qdrant-backed vector store."""

from __future__ import annotations

import asyncio
from pathlib import Path
import uuid
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from backend.config.settings import Settings
from backend.core.errors import ConfigurationError
from backend.repointel.models import CodeChunk, RetrievalHit


class QdrantVectorStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        Path(settings.vector_db_path).mkdir(parents=True, exist_ok=True)
        self._qdrant_client: QdrantClient | None = None
        self._initialized = False

    async def upsert_chunks(self, chunks: list[CodeChunk], vectors: dict[str, list[float]]) -> None:
        if not chunks:
            return
        first_vector = vectors[chunks[0].id]
        await self._ensure_collection(len(first_vector))
        points = []
        for chunk in chunks:
            points.append(
                qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors[chunk.id],
                    payload={
                        "file_path": chunk.file_path,
                        "symbol_id": chunk.symbol_id,
                        "symbol_name": chunk.symbol_name,
                        "language": chunk.language.value,
                        "content": chunk.content,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        **chunk.metadata,
                    },
                )
            )
        await asyncio.to_thread(
            self._get_client().upsert,
            self._settings.vector_collection,
            points,
        )

    async def delete_paths(self, file_paths: list[str]) -> None:
        if not file_paths or not self._initialized:
            return
        for file_path in file_paths:
            await asyncio.to_thread(
                self._get_client().delete,
                self._settings.vector_collection,
                qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="file_path",
                                match=qdrant_models.MatchValue(value=file_path),
                            )
                        ]
                    )
                ),
            )

    async def query(self, vector: list[float], limit: int) -> list[RetrievalHit]:
        await self._ensure_collection(len(vector))
        response = await asyncio.to_thread(
            self._get_client().query_points,
            self._settings.vector_collection,
            vector,
            limit=limit,
            with_payload=True,
        )
        hits: list[RetrievalHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                RetrievalHit(
                    chunk_id=str(point.id),
                    file_path=str(payload.get("file_path", "")),
                    symbol_name=payload.get("symbol_name"),
                    language=payload.get("language", "unknown"),
                    score=float(point.score or 0.0),
                    vector_score=float(point.score or 0.0),
                    content=str(payload.get("content", "")),
                    metadata=dict(payload),
                )
            )
        return hits

    async def healthcheck(self) -> dict[str, object]:
        try:
            collections = await asyncio.to_thread(self._get_client().get_collections)
        except Exception as exc:
            raise ConfigurationError(
                f"Qdrant connectivity check failed for local path '{self._settings.vector_db_path}'."
            ) from exc
        return {
            "vector_db_path": self._settings.vector_db_path,
            "collections": len(collections.collections),
            "collection_name": self._settings.vector_collection,
        }

    async def close(self) -> None:
        if self._qdrant_client is not None:
            await asyncio.to_thread(self._qdrant_client.close)
            self._qdrant_client = None
            self._initialized = False

    async def _ensure_collection(self, vector_size: int) -> None:
        if self._initialized:
            return
        exists = await asyncio.to_thread(
            self._get_client().collection_exists,
            self._settings.vector_collection,
        )
        if not exists:
            await asyncio.to_thread(
                self._get_client().create_collection,
                self._settings.vector_collection,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
        self._initialized = True

    def _get_client(self) -> QdrantClient:
        if self._qdrant_client is None:
            try:
                self._qdrant_client = QdrantClient(path=self._settings.vector_db_path)
            except Exception as exc:
                raise ConfigurationError(
                    f"Failed to initialize local Qdrant client at '{self._settings.vector_db_path}'."
                ) from exc
        return self._qdrant_client
