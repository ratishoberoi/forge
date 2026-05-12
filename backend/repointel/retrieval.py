"""Hybrid retrieval over vector and lexical indices."""

from __future__ import annotations

import math
import re
from collections import Counter

from backend.config.settings import Settings, get_settings
from backend.repointel.embeddings import EmbeddingService
from backend.repointel.models import CodeChunk, RetrievalHit
from backend.repointel.vector_store import QdrantVectorStore


class BM25Index:
    def __init__(self) -> None:
        self._documents: dict[str, CodeChunk] = {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_freqs: dict[str, int] = {}
        self._avg_doc_len = 0.0

    def rebuild(self, chunks: list[CodeChunk]) -> None:
        self._documents = {chunk.id: chunk for chunk in chunks}
        self._doc_tokens = {chunk.id: self._tokenize(chunk.content) for chunk in chunks}
        self._doc_freqs = {}
        total_len = 0
        for tokens in self._doc_tokens.values():
            total_len += len(tokens)
            for token in set(tokens):
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
        self._avg_doc_len = total_len / max(len(self._doc_tokens), 1)

    def search(self, query: str, limit: int) -> list[RetrievalHit]:
        query_tokens = self._tokenize(query)
        hits: list[RetrievalHit] = []
        for chunk_id, tokens in self._doc_tokens.items():
            score = self._score(query_tokens, tokens)
            if score <= 0:
                continue
            chunk = self._documents[chunk_id]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.id,
                    file_path=chunk.file_path,
                    symbol_name=chunk.symbol_name,
                    language=chunk.language,
                    score=score,
                    bm25_score=score,
                    content=chunk.content,
                    metadata=chunk.metadata,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        k1 = 1.5
        b = 0.75
        frequencies = Counter(doc_tokens)
        score = 0.0
        for token in query_tokens:
            df = self._doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(self._doc_tokens) - df + 0.5) / (df + 0.5))
            freq = frequencies[token]
            denom = freq + k1 * (1 - b + b * len(doc_tokens) / max(self._avg_doc_len, 1.0))
            score += idf * ((freq * (k1 + 1)) / denom)
        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


class HybridRetrievalEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embeddings = embeddings or EmbeddingService(self._settings)
        self._vector_store = vector_store or QdrantVectorStore(self._settings)
        self._bm25 = BM25Index()
        self._chunks: dict[str, CodeChunk] = {}

    def rebuild_lexical_index(self, chunks: list[CodeChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        self._bm25.rebuild(list(self._chunks.values()))

    def remove_paths(self, file_paths: list[str]) -> None:
        remove_set = set(file_paths)
        self._chunks = {
            chunk_id: chunk for chunk_id, chunk in self._chunks.items() if chunk.file_path not in remove_set
        }
        self._bm25.rebuild(list(self._chunks.values()))

    async def retrieve(self, query: str, limit: int | None = None) -> list[RetrievalHit]:
        limit = limit or self._settings.repo_retrieval_limit
        query_vector = (await self._embeddings.embed_texts({"query": query}))["query"]
        vector_hits = await self._vector_store.query(query_vector, limit=limit * 2)
        lexical_hits = self._bm25.search(query, limit=limit * 2)
        merged = self._merge_hits(vector_hits, lexical_hits)
        reranked = self._rerank(query, merged)
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        return reranked[:limit]

    async def healthcheck(self) -> dict[str, object]:
        query_vector = await self._embeddings.embed_text("forge retrieval healthcheck", cache_key="retrieval-healthcheck")
        hits = await self._vector_store.query(query_vector, limit=1)
        return {
            "query_vector_dimension": len(query_vector),
            "lexical_documents": len(self._chunks),
            "vector_hits": len(hits),
        }

    async def validate_startup(self) -> dict[str, object]:
        return await self.healthcheck()

    def _merge_hits(self, vector_hits: list[RetrievalHit], lexical_hits: list[RetrievalHit]) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        for hit in vector_hits + lexical_hits:
            existing = merged.get(hit.chunk_id)
            if existing is None:
                merged[hit.chunk_id] = hit
                continue
            existing.vector_score = max(existing.vector_score, hit.vector_score)
            existing.bm25_score = max(existing.bm25_score, hit.bm25_score)
            existing.score = max(existing.score, hit.score)
        return list(merged.values())

    def _rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        query_terms = set(BM25Index._tokenize(query))
        for hit in hits:
            content_terms = set(BM25Index._tokenize(hit.content))
            overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            hit.rerank_score = overlap
            hit.score = (hit.vector_score * 0.5) + (hit.bm25_score * 0.35) + (hit.rerank_score * 0.15)
        return hits
