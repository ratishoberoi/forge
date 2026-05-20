from __future__ import annotations

import hashlib
import json
import math
import shutil
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SemanticMemoryItem:
    item_id: str
    repository_path: str
    kind: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "repository_path": self.repository_path,
            "kind": self.kind,
            "text": self.text,
            "metadata": self.metadata,
            "score": round(self.score, 4),
            "created_at": self.created_at,
        }


class LocalHashEmbedder:
    """Deterministic local embedding fallback for offline workspaces."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class SemanticMemory:
    """
    Local semantic memory backed by SQLite.
    The vector representation is local-only and deterministic; no cloud inference is required.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        qdrant_path: str | None = None,
        embedder: LocalHashEmbedder | None = None,
    ) -> None:
        self.path = Path(path or ".forge/semantic_memory.sqlite3").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.qdrant_path = Path(qdrant_path or ".forge/qdrant").resolve()
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or LocalHashEmbedder()
        self._conn = self._connect()
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def upsert(
        self,
        *,
        repository_path: str,
        kind: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        item_id: str | None = None,
    ) -> SemanticMemoryItem:
        resolved_repo = str(Path(repository_path).resolve())
        normalized_text = text.strip()
        stable_id = item_id or hashlib.sha256(
            f"{resolved_repo}\0{kind}\0{normalized_text}".encode("utf-8")
        ).hexdigest()
        created_at = datetime.now(timezone.utc).isoformat()
        vector = self.embedder.embed(normalized_text)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (item_id, repository_path, kind, text, metadata, vector, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id,
                resolved_repo,
                kind,
                normalized_text,
                json.dumps(metadata or {}, sort_keys=True),
                json.dumps(vector),
                created_at,
            ),
        )
        self._conn.commit()
        return SemanticMemoryItem(
            item_id=stable_id,
            repository_path=resolved_repo,
            kind=kind,
            text=normalized_text,
            metadata=metadata or {},
            created_at=created_at,
        )

    def retrieve(
        self,
        *,
        repository_path: str,
        query: str,
        kinds: list[str] | None = None,
        limit: int = 8,
    ) -> list[SemanticMemoryItem]:
        resolved_repo = str(Path(repository_path).resolve())
        query_vector = self.embedder.embed(query)
        rows = self._rows(repository_path=resolved_repo, kinds=kinds)
        scored: list[SemanticMemoryItem] = []
        query_terms = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
        for row in rows:
            try:
                vector = json.loads(str(row["vector"]))
                metadata = json.loads(str(row["metadata"]))
            except json.JSONDecodeError:
                self._delete_item(str(row["item_id"]))
                print(
                    "[MEMORY_CORRUPTION_RECOVERED] semantic_memory "
                    f"item_id={row['item_id']} repository={resolved_repo}"
                )
                continue
            semantic_score = _cosine(query_vector, vector)
            lexical_score = _lexical_score(query_terms, str(row["text"]))
            score = semantic_score * 0.75 + lexical_score * 0.25
            scored.append(
                SemanticMemoryItem(
                    item_id=str(row["item_id"]),
                    repository_path=str(row["repository_path"]),
                    kind=str(row["kind"]),
                    text=str(row["text"]),
                    metadata=metadata,
                    score=score,
                    created_at=str(row["created_at"]),
                )
            )
        return sorted(scored, key=lambda item: (-item.score, item.created_at), reverse=False)[:limit]

    def list_recent(self, *, repository_path: str | None = None, limit: int = 20) -> list[SemanticMemoryItem]:
        params: tuple[Any, ...]
        where = ""
        if repository_path:
            where = "WHERE repository_path = ?"
            params = (str(Path(repository_path).resolve()), limit)
        else:
            params = (limit,)
        cursor = self._conn.execute(
            f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        return [self._item_from_row(row) for row in cursor.fetchall()]

    def stats(self, *, repository_path: str | None = None) -> dict[str, Any]:
        params: tuple[Any, ...] = ()
        where = ""
        if repository_path:
            where = "WHERE repository_path = ?"
            params = (str(Path(repository_path).resolve()),)
        count = self._conn.execute(f"SELECT COUNT(*) FROM memories {where}", params).fetchone()[0]
        kinds = self._conn.execute(
            f"SELECT kind, COUNT(*) FROM memories {where} GROUP BY kind",
            params,
        ).fetchall()
        return {
            "memory_path": str(self.path),
            "qdrant_path": str(self.qdrant_path),
            "embedding": "local-hash-fallback",
            "items": int(count),
            "kinds": {str(kind): int(total) for kind, total in kinds},
        }

    def delete_repository_kind(self, *, repository_path: str, kind: str) -> None:
        self._conn.execute(
            "DELETE FROM memories WHERE repository_path = ? AND kind = ?",
            (str(Path(repository_path).resolve()), kind),
        )
        self._conn.commit()

    def _rows(self, *, repository_path: str, kinds: list[str] | None = None) -> list[sqlite3.Row]:
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            cursor = self._conn.execute(
                f"SELECT * FROM memories WHERE repository_path = ? AND kind IN ({placeholders})",
                (repository_path, *kinds),
            )
        else:
            cursor = self._conn.execute("SELECT * FROM memories WHERE repository_path = ?", (repository_path,))
        return list(cursor.fetchall())

    @staticmethod
    def _item_from_row(row: sqlite3.Row | tuple[Any, ...]) -> SemanticMemoryItem:
        if not isinstance(row, sqlite3.Row):
            raise TypeError("semantic memory row factory was not configured")
        try:
            metadata = json.loads(str(row["metadata"]))
        except json.JSONDecodeError:
            metadata = {"corrupt_metadata": True}
        return SemanticMemoryItem(
            item_id=str(row["item_id"]),
            repository_path=str(row["repository_path"]),
            kind=str(row["kind"]),
            text=str(row["text"]),
            metadata=metadata,
            created_at=str(row["created_at"]),
        )

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.path, check_same_thread=False)
            check = connection.execute("PRAGMA integrity_check").fetchone()
            if check and str(check[0]).lower() != "ok":
                raise sqlite3.DatabaseError(str(check[0]))
            return connection
        except sqlite3.DatabaseError as exc:
            self._quarantine_database(str(exc))
            return sqlite3.connect(self.path, check_same_thread=False)

    def _initialize_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                item_id TEXT PRIMARY KEY,
                repository_path TEXT NOT NULL,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                vector TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_repo ON memories(repository_path)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
        self._conn.commit()

    def _delete_item(self, item_id: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE item_id = ?", (item_id,))
        self._conn.commit()

    def _quarantine_database(self, reason: str) -> None:
        if not self.path.exists():
            return
        quarantine = self.path.with_name(
            f"{self.path.name}.corrupt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.bak"
        )
        try:
            shutil.copy2(self.path, quarantine)
            self.path.unlink()
        except OSError:
            pass
        print(
            "[MEMORY_CORRUPTION_RECOVERED] semantic_memory "
            f"path={self.path} backup={quarantine} reason={reason}"
        )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(left * right for left, right in zip(a, b, strict=False))


def _lexical_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
    return len(query_terms & text_terms) / len(query_terms)
