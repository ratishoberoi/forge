"""Language helpers for tree-sitter-backed parsing."""

from __future__ import annotations

from backend.repointel.models import Language

TREE_SITTER_LANGUAGE_SOURCES = {
    Language.PYTHON: ("tree_sitter_python", "language"),
    Language.TYPESCRIPT: ("tree_sitter_typescript", "language_typescript"),
    Language.JAVASCRIPT: ("tree_sitter_javascript", "language"),
    Language.GO: ("tree_sitter_go", "language"),
    Language.RUST: ("tree_sitter_rust", "language"),
}
