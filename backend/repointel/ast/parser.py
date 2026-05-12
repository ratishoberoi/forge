"""Tree-sitter parsing and symbol extraction."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any

from tree_sitter import Language as TreeSitterLanguage
from tree_sitter import Parser

from backend.core.errors import ConfigurationError
from backend.config.settings import Settings, get_settings
from backend.repointel.ast.languages import TREE_SITTER_LANGUAGE_SOURCES
from backend.repointel.models import (
    CodeSymbol,
    ImportRecord,
    Language,
    ParsedFile,
    RepositoryFile,
    SymbolKind,
    SymbolReference,
)

_SYMBOL_NODE_TYPES = {
    Language.PYTHON: {
        "class_definition": SymbolKind.CLASS,
        "function_definition": SymbolKind.FUNCTION,
    },
    Language.TYPESCRIPT: {
        "class_declaration": SymbolKind.CLASS,
        "function_declaration": SymbolKind.FUNCTION,
        "method_definition": SymbolKind.METHOD,
    },
    Language.JAVASCRIPT: {
        "class_declaration": SymbolKind.CLASS,
        "function_declaration": SymbolKind.FUNCTION,
        "method_definition": SymbolKind.METHOD,
    },
    Language.GO: {
        "function_declaration": SymbolKind.FUNCTION,
        "method_declaration": SymbolKind.METHOD,
        "type_declaration": SymbolKind.CLASS,
    },
    Language.RUST: {
        "function_item": SymbolKind.FUNCTION,
        "impl_item": SymbolKind.CLASS,
        "struct_item": SymbolKind.CLASS,
    },
}


@dataclass(slots=True)
class LanguageParser:
    language: TreeSitterLanguage
    parser: Parser


class TreeSitterAstEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._parsers: dict[Language, LanguageParser] = {}

    def parse(self, repo_file: RepositoryFile) -> ParsedFile:
        parser = self._get_parser(repo_file.language)
        tree = parser.parser.parse(repo_file.content.encode("utf-8"))
        root = tree.root_node
        symbols = self._extract_symbols(root, repo_file)
        imports = self._extract_imports(repo_file)
        exports = self._extract_exports(repo_file)
        references = self._extract_references(repo_file.content)
        return ParsedFile(
            file=repo_file,
            symbols=symbols,
            imports=imports,
            exports=exports,
            references=references,
        )

    def _get_parser(self, language: Language) -> LanguageParser:
        if language in self._parsers:
            return self._parsers[language]
        source = TREE_SITTER_LANGUAGE_SOURCES.get(language)
        if not source:
            raise ConfigurationError(f"Unsupported language for tree-sitter parsing: {language}.")
        package_name, attr_name = source
        try:
            module = importlib.import_module(package_name)
        except ModuleNotFoundError as exc:
            raise ConfigurationError(
                f"Missing tree-sitter language package '{package_name}'. Install repository intelligence dependencies."
            ) from exc
        language_factory = getattr(module, attr_name, None)
        if not callable(language_factory):
            raise ConfigurationError(
                f"Tree-sitter package '{package_name}' does not expose callable '{attr_name}()'."
            )
        try:
            language_obj = TreeSitterLanguage(language_factory())
        except Exception as exc:
            raise ConfigurationError(
                f"Failed to initialize tree-sitter language '{language.value}' from package '{package_name}'. "
                "Expected a tree_sitter.Language-compatible binding."
            ) from exc
        if not isinstance(language_obj, TreeSitterLanguage):
            raise ConfigurationError(
                f"Tree-sitter language loader for '{language.value}' returned invalid type "
                f"'{type(language_obj).__name__}'."
            )
        parser = Parser()
        try:
            parser.language = language_obj
        except Exception as exc:
            raise ConfigurationError(
                f"Tree-sitter parser rejected language '{language.value}'. Parser setup failed."
            ) from exc
        loaded = LanguageParser(language=language_obj, parser=parser)
        self._parsers[language] = loaded
        return loaded

    def validate_languages(self, languages: list[Language] | None = None) -> dict[Language, str]:
        languages = languages or [
            Language.PYTHON,
            Language.JAVASCRIPT,
            Language.TYPESCRIPT,
            Language.GO,
            Language.RUST,
        ]
        results: dict[Language, str] = {}
        for language in languages:
            parser = self._get_parser(language)
            if not isinstance(parser.language, TreeSitterLanguage):
                raise ConfigurationError(
                    f"Tree-sitter language '{language.value}' did not initialize to a Language object."
                )
            results[language] = parser.language.name or language.value
        return results

    def _extract_symbols(self, root: Any, repo_file: RepositoryFile) -> list[CodeSymbol]:
        symbols: list[CodeSymbol] = []
        node_type_map = _SYMBOL_NODE_TYPES.get(repo_file.language, {})

        def walk(node: Any, parent_symbol: str | None = None) -> None:
            kind = node_type_map.get(node.type)
            current_parent = parent_symbol
            if kind is not None:
                name_node = node.child_by_field_name("name")
                symbol_name = self._node_text(name_node, repo_file.content) if name_node else node.type
                symbol_id = f"{repo_file.path}:{symbol_name}:{node.start_point[0] + 1}"
                symbol_kind = kind
                if repo_file.language is Language.PYTHON and kind is SymbolKind.FUNCTION and parent_symbol:
                    symbol_kind = SymbolKind.METHOD
                symbols.append(
                    CodeSymbol(
                        id=symbol_id,
                        name=symbol_name,
                        kind=symbol_kind,
                        language=repo_file.language,
                        file_path=repo_file.path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent_symbol=parent_symbol,
                        signature=self._node_text(node, repo_file.content).splitlines()[0][:240],
                        docstring=self._extract_docstring(node, repo_file.content, repo_file.language),
                        references=self._extract_symbol_references(node, repo_file.content),
                    )
                )
                current_parent = symbol_id
            for child in node.children:
                walk(child, current_parent)

        walk(root)
        return symbols

    def _extract_imports(self, repo_file: RepositoryFile) -> list[ImportRecord]:
        imports: list[ImportRecord] = []
        for line_number, line in enumerate(repo_file.content.splitlines(), start=1):
            stripped = line.strip()
            if repo_file.language is Language.PYTHON:
                if stripped.startswith("import "):
                    imports.append(ImportRecord(module=stripped.removeprefix("import ").strip(), line=line_number))
                elif stripped.startswith("from "):
                    match = re.match(r"from\s+([^\s]+)\s+import\s+(.+)", stripped)
                    if match:
                        imports.append(
                            ImportRecord(
                                module=match.group(1),
                                symbols=[part.strip() for part in match.group(2).split(",")],
                                line=line_number,
                            )
                        )
            elif repo_file.language in {Language.JAVASCRIPT, Language.TYPESCRIPT}:
                if stripped.startswith("import "):
                    module = stripped.split(" from ")[-1].strip(";'\"")
                    imports.append(ImportRecord(module=module, line=line_number))
            elif repo_file.language is Language.GO and stripped.startswith("import"):
                imports.append(ImportRecord(module=stripped, line=line_number))
            elif repo_file.language is Language.RUST and stripped.startswith("use "):
                imports.append(ImportRecord(module=stripped.removeprefix("use ").strip(" ;"), line=line_number))
        return imports

    def _extract_exports(self, repo_file: RepositoryFile) -> list[str]:
        exports: list[str] = []
        if repo_file.language not in {Language.JAVASCRIPT, Language.TYPESCRIPT}:
            return exports
        for line in repo_file.content.splitlines():
            stripped = line.strip()
            if stripped.startswith("export "):
                exports.append(stripped)
        return exports

    def _extract_references(self, content: str) -> list[SymbolReference]:
        references: list[SymbolReference] = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", line):
                references.append(SymbolReference(name=match.group(1), line=line_number))
        return references

    def _extract_symbol_references(self, node: Any, content: str) -> list[SymbolReference]:
        text = self._node_text(node, content)
        references: list[SymbolReference] = []
        for offset, line in enumerate(text.splitlines(), start=node.start_point[0] + 1):
            for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", line):
                references.append(SymbolReference(name=match.group(1), line=offset))
        return references

    def _extract_docstring(self, node: Any, content: str, language: Language) -> str | None:
        if language is not Language.PYTHON:
            return None
        body = node.child_by_field_name("body")
        if body is None or not body.children:
            return None
        first_child = body.children[0]
        text = self._node_text(first_child, content).strip()
        if text.startswith('"""') or text.startswith("'''"):
            return text.strip("\"'")
        return None

    @staticmethod
    def _node_text(node: Any, content: str) -> str:
        if node is None:
            return ""
        return content[node.start_byte : node.end_byte]


TreeSitterParser = TreeSitterAstEngine

__all__ = ["LanguageParser", "TreeSitterAstEngine", "TreeSitterParser"]
