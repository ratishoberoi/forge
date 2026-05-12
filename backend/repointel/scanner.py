"""Repository scanning with ignore and incremental indexing support."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path

from backend.config.settings import Settings
from backend.repointel.models import Language, RepositoryFile, RepositoryScanResult

_LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".go": Language.GO,
    ".rs": Language.RUST,
}


class RepositoryScanner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def scan(self, root: str) -> RepositoryScanResult:
        root_path = Path(root).resolve()
        previous_manifest = self.load_manifest()
        gitignore_patterns = self._load_gitignore_patterns(root_path)

        files: list[RepositoryFile] = []
        manifest: dict[str, str] = {}
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            if path.stat().st_size > self._settings.repo_max_file_bytes:
                continue

            relative_path = path.relative_to(root_path).as_posix()
            if self._should_ignore(relative_path, gitignore_patterns):
                continue

            language = _LANGUAGE_MAP.get(path.suffix.lower(), Language.UNKNOWN)
            if language is Language.UNKNOWN:
                continue

            content = path.read_text(encoding="utf-8", errors="ignore")
            sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            manifest[relative_path] = sha256
            if self._settings.repo_incremental and previous_manifest.get(relative_path) == sha256:
                continue

            stat = path.stat()
            files.append(
                RepositoryFile(
                    path=relative_path,
                    absolute_path=str(path),
                    language=language,
                    sha256=sha256,
                    size_bytes=stat.st_size,
                    modified_at=stat.st_mtime,
                    content=content,
                )
            )

        deleted_paths = sorted(set(previous_manifest) - set(manifest))
        return RepositoryScanResult(
            root=str(root_path),
            files=files,
            deleted_paths=deleted_paths,
            manifest=manifest,
        )

    def load_manifest(self) -> dict[str, str]:
        state_path = Path(self._settings.repo_index_state_path)
        if not state_path.exists():
            return {}
        return json.loads(state_path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: dict[str, str]) -> None:
        state_path = Path(self._settings.repo_index_state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _should_ignore(self, relative_path: str, gitignore_patterns: list[str]) -> bool:
        parts = relative_path.split("/")
        patterns = list(self._settings.repo_ignore_patterns) + gitignore_patterns
        return any(
            fnmatch.fnmatch(relative_path, pattern)
            or any(fnmatch.fnmatch(part, pattern) for part in parts)
            for pattern in patterns
        )

    def _load_gitignore_patterns(self, root_path: Path) -> list[str]:
        if not self._settings.repo_respect_gitignore:
            return []
        gitignore = root_path / ".gitignore"
        if not gitignore.exists():
            return []
        patterns: list[str] = []
        for line in gitignore.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                patterns.append(stripped)
        return patterns
