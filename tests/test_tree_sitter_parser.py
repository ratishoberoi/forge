from __future__ import annotations

from backend.repointel.ast.parser import TreeSitterAstEngine, TreeSitterParser
from backend.repointel.models import Language, RepositoryFile


def test_tree_sitter_language_initialization_for_supported_languages() -> None:
    engine = TreeSitterAstEngine()
    validated = engine.validate_languages(
        [
            Language.PYTHON,
            Language.JAVASCRIPT,
            Language.TYPESCRIPT,
            Language.GO,
            Language.RUST,
        ]
    )
    assert set(validated) == {
        Language.PYTHON,
        Language.JAVASCRIPT,
        Language.TYPESCRIPT,
        Language.GO,
        Language.RUST,
    }


def test_tree_sitter_parses_python_file() -> None:
    engine = TreeSitterParser()
    parsed = engine.parse(
        RepositoryFile(
            path="app.py",
            absolute_path="/tmp/app.py",
            language=Language.PYTHON,
            sha256="x",
            size_bytes=10,
            modified_at=0.0,
            content="class App:\n    def run(self):\n        return 1\n",
        )
    )
    symbol_names = [symbol.name for symbol in parsed.symbols]
    assert "App" in symbol_names
    assert "run" in symbol_names
