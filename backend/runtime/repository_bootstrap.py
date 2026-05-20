from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.runtime.framework_detection import FrameworkDetector, FrameworkProfile
from backend.runtime.patch_writer import PatchResult, PatchWriter
from backend.runtime.repo_workspace import RepositoryWorkspace


@dataclass(slots=True)
class BootstrapResult:
    applied: bool
    framework: FrameworkProfile
    files: list[str] = field(default_factory=list)
    results: list[PatchResult] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "framework": self.framework.to_dict(),
            "files": self.files,
            "results": [
                {
                    "file_path": result.file_path,
                    "success": result.success,
                    "resolved_path": str(result.resolved_path) if result.resolved_path else None,
                    "error": result.error,
                }
                for result in self.results
            ],
            "reason": self.reason,
        }


class RepositoryBootstrap:
    """Creates starter project structure for empty repositories before model execution."""

    APP_MARKERS = {
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "app.py",
        "main.py",
        "manage.py",
        "index.html",
        "src",
        "app",
    }

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.detector = FrameworkDetector()
        self.writer = PatchWriter(RepositoryWorkspace(str(self.repo_root)), backup=False)

    def bootstrap_if_needed(self, objective: str) -> BootstrapResult:
        existing = {path.name for path in self.repo_root.iterdir()} if self.repo_root.exists() else set()
        detected = self.detector.detect(self.repo_root)
        if existing & self.APP_MARKERS:
            return BootstrapResult(
                applied=False,
                framework=detected,
                reason="application markers already present",
            )
        profile = self.detector.profile_for_objective(objective)
        files = self._files_for(objective, profile)
        results = self.writer.apply_many(files)
        return BootstrapResult(
            applied=True,
            framework=profile,
            files=list(files),
            results=results,
            reason="empty repository bootstrapped",
        )

    def _files_for(self, objective: str, profile: FrameworkProfile) -> dict[str, str]:
        if "React" in profile.frameworks or "Vite" in profile.frameworks:
            return self._vite_files(objective)
        if "FastAPI" in profile.frameworks:
            return self._fastapi_files(objective)
        return self._python_files(objective)

    @staticmethod
    def _python_files(objective: str) -> dict[str, str]:
        return {
            "pyproject.toml": (
                "[project]\nname = \"forge-app\"\nversion = \"0.1.0\"\n\n"
                "[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\n"
            ),
            "app.py": (
                "def main() -> str:\n"
                f"    return {objective!r}\n\n"
                "if __name__ == \"__main__\":\n"
                "    print(main())\n"
            ),
            "tests/test_app.py": (
                "from app import main\n\n"
                "def test_main_returns_objective():\n"
                "    assert main()\n"
            ),
            "README.md": f"# Forge App\n\nBootstrapped for objective: {objective}\n",
        }

    @staticmethod
    def _fastapi_files(objective: str) -> dict[str, str]:
        return {
            "pyproject.toml": (
                "[project]\nname = \"forge-api\"\nversion = \"0.1.0\"\n"
                "dependencies = [\"fastapi\", \"uvicorn\"]\n\n"
                "[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\n"
            ),
            "app/__init__.py": "# Forge API package\n",
            "app/main.py": (
                "try:\n"
                "    from fastapi import FastAPI\n"
                "except Exception:  # pragma: no cover\n"
                "    FastAPI = None\n\n"
                "def create_app():\n"
                "    if FastAPI is None:\n"
                "        return None\n"
                "    app = FastAPI(title=\"Forge API\")\n\n"
                "    @app.get('/health')\n"
                "    def health():\n"
                "        return {'status': 'ok'}\n\n"
                "    return app\n\n"
                "app = create_app()\n\n"
                "def objective_summary() -> str:\n"
                f"    return {objective!r}\n"
            ),
            "tests/test_app.py": (
                "from app.main import objective_summary\n\n"
                "def test_objective_summary():\n"
                "    assert objective_summary()\n"
            ),
            "README.md": f"# Forge API\n\nBootstrapped for objective: {objective}\n\nRun tests with `pytest -q`.\n",
            "Dockerfile": "FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nCMD [\"python\", \"-m\", \"app.main\"]\n",
        }

    @staticmethod
    def _vite_files(objective: str) -> dict[str, str]:
        return {
            "package.json": (
                "{\n"
                "  \"name\": \"forge-web-app\",\n"
                "  \"version\": \"0.1.0\",\n"
                "  \"private\": true,\n"
                "  \"type\": \"module\",\n"
                "  \"scripts\": {\n"
                "    \"test\": \"node --test tests/app.test.mjs\",\n"
                "    \"build\": \"node scripts/build.mjs\"\n"
                "  },\n"
                "  \"dependencies\": {\"@vitejs/plugin-react\": \"latest\", \"vite\": \"latest\", \"react\": \"latest\", \"react-dom\": \"latest\"},\n"
                "  \"devDependencies\": {}\n"
                "}\n"
            ),
            "index.html": "<!doctype html><html><head><title>Forge App</title></head><body><div id=\"root\"></div><script type=\"module\" src=\"/src/main.jsx\"></script></body></html>\n",
            "src/main.jsx": (
                "import React from 'react';\n"
                "import { createRoot } from 'react-dom/client';\n"
                "import { App } from './App.jsx';\n\n"
                "createRoot(document.getElementById('root')).render(<App />);\n"
            ),
            "src/App.jsx": (
                "export function App() {\n"
                "  return <main><h1>Forge App</h1><p>Production starter workspace.</p></main>;\n"
                "}\n\n"
                "export const objective = " + repr(objective) + ";\n"
            ),
            "tests/app.test.mjs": (
                "import test from 'node:test';\n"
                "import assert from 'node:assert/strict';\n"
                "import { readFileSync } from 'node:fs';\n\n"
                "test('application source exists', () => {\n"
                "  const source = readFileSync('src/App.jsx', 'utf8');\n"
                "  assert.match(source, /Forge App/);\n"
                "});\n"
            ),
            "scripts/build.mjs": (
                "import { mkdirSync, copyFileSync } from 'node:fs';\n"
                "mkdirSync('dist', { recursive: true });\n"
                "copyFileSync('index.html', 'dist/index.html');\n"
            ),
            "README.md": f"# Forge Web App\n\nBootstrapped for objective: {objective}\n\nRun `npm test` and `npm run build`.\n",
        }
