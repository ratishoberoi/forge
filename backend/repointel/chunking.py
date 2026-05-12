"""AST-aware chunking based on extracted symbols."""

from __future__ import annotations

import hashlib

from backend.repointel.models import CodeChunk, ParsedFile


class AstAwareChunker:
    def chunk(self, parsed: ParsedFile) -> list[CodeChunk]:
        import_lines = [record.module for record in parsed.imports]
        source_lines = parsed.file.content.splitlines()
        chunks: list[CodeChunk] = []

        for symbol in parsed.symbols:
            start = max(symbol.start_line - 1, 0)
            end = min(symbol.end_line, len(source_lines))
            symbol_content = "\n".join(source_lines[start:end]).strip()
            if not symbol_content:
                continue
            chunk_text = symbol_content
            if import_lines:
                chunk_text = "\n".join(import_lines) + "\n\n" + symbol_content
            chunks.append(
                CodeChunk(
                    id=self._chunk_id(parsed.file.path, symbol.id),
                    file_path=parsed.file.path,
                    language=parsed.file.language,
                    symbol_id=symbol.id,
                    symbol_name=symbol.name,
                    content=chunk_text,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    imports=import_lines,
                    metadata={
                        "symbol_kind": symbol.kind.value,
                        "docstring": symbol.docstring,
                    },
                )
            )

        if not chunks:
            chunks.append(
                CodeChunk(
                    id=self._chunk_id(parsed.file.path, "module"),
                    file_path=parsed.file.path,
                    language=parsed.file.language,
                    symbol_id=None,
                    symbol_name=None,
                    content=parsed.file.content,
                    start_line=1,
                    end_line=len(source_lines),
                    imports=import_lines,
                    metadata={"symbol_kind": "module"},
                )
            )
        return chunks

    @staticmethod
    def _chunk_id(file_path: str, symbol_id: str) -> str:
        return hashlib.sha256(f"{file_path}:{symbol_id}".encode("utf-8")).hexdigest()
