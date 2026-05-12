"""Typed models for repository intelligence."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Language(StrEnum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


class SymbolKind(StrEnum):
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    IMPORT = "import"
    EXPORT = "export"
    MODULE = "module"


class RepositoryFile(BaseModel):
    path: str
    absolute_path: str
    language: Language
    sha256: str
    size_bytes: int
    modified_at: float
    content: str


class ImportRecord(BaseModel):
    module: str
    symbols: list[str] = Field(default_factory=list)
    line: int
    is_export: bool = False


class SymbolReference(BaseModel):
    name: str
    line: int
    kind: str = "identifier"


class CodeSymbol(BaseModel):
    id: str
    name: str
    kind: SymbolKind
    language: Language
    file_path: str
    start_line: int
    end_line: int
    parent_symbol: str | None = None
    signature: str | None = None
    docstring: str | None = None
    references: list[SymbolReference] = Field(default_factory=list)


class ParsedFile(BaseModel):
    file: RepositoryFile
    symbols: list[CodeSymbol] = Field(default_factory=list)
    imports: list[ImportRecord] = Field(default_factory=list)
    exports: list[str] = Field(default_factory=list)
    references: list[SymbolReference] = Field(default_factory=list)


class CodeChunk(BaseModel):
    id: str
    file_path: str
    language: Language
    symbol_id: str | None = None
    symbol_name: str | None = None
    content: str
    start_line: int
    end_line: int
    imports: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RepositoryScanResult(BaseModel):
    root: str
    files: list[RepositoryFile]
    deleted_paths: list[str] = Field(default_factory=list)
    manifest: dict[str, str] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    chunk_id: str
    file_path: str
    symbol_name: str | None = None
    language: Language
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPackage(BaseModel):
    query: str
    hits: list[RetrievalHit]
    related_symbols: list[CodeSymbol]
    related_files: list[str]
    dependency_neighbors: dict[str, list[str]]


class PlanStep(BaseModel):
    title: str
    description: str
    file_paths: list[str]
    impact: str


class ExecutionPlan(BaseModel):
    query: str
    impacted_files: list[str]
    dependency_risks: list[str]
    steps: list[PlanStep]


class IndexingStats(BaseModel):
    files_scanned: int = 0
    files_indexed: int = 0
    symbols_extracted: int = 0
    chunks_indexed: int = 0
