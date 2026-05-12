from __future__ import annotations

import asyncio
from pathlib import Path

from backend.config.settings import Settings
from backend.repointel.scanner import RepositoryScanner


def test_repository_scanner_respects_ignore_patterns(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (root / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "ignored.py").write_text("print('no')\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "skip.js").write_text("console.log('x')\n", encoding="utf-8")

    settings = Settings(
        repo_index_state_path=str(tmp_path / "index.json"),
        repo_incremental=False,
    )
    scanner = RepositoryScanner(settings)
    result = asyncio.run(scanner.scan(str(root)))

    assert [file.path for file in result.files] == ["keep.py"]


def test_repository_scanner_incremental_manifest(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    file_path = root / "main.py"
    file_path.write_text("print('one')\n", encoding="utf-8")

    settings = Settings(
        repo_index_state_path=str(tmp_path / "index.json"),
        repo_incremental=True,
    )
    scanner = RepositoryScanner(settings)
    first = asyncio.run(scanner.scan(str(root)))
    scanner.save_manifest(first.manifest)
    second = asyncio.run(scanner.scan(str(root)))

    assert len(first.files) == 1
    assert len(second.files) == 0
