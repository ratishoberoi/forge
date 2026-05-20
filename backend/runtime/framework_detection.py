from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FrameworkProfile:
    language: str
    frameworks: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    databases: list[str] = field(default_factory=list)
    infrastructure: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)
    build_commands: list[str] = field(default_factory=list)
    conventions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "frameworks": self.frameworks,
            "package_managers": self.package_managers,
            "databases": self.databases,
            "infrastructure": self.infrastructure,
            "test_commands": self.test_commands,
            "build_commands": self.build_commands,
            "conventions": self.conventions,
        }


class FrameworkDetector:
    """Detects framework conventions from repository files without network access."""

    def detect(self, repo_root: str | Path) -> FrameworkProfile:
        root = Path(repo_root).resolve()
        root_files = {path.name for path in root.iterdir() if path.is_file()}
        paths = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
        package_json = self._package_json(root)
        pyproject = self._read(root / "pyproject.toml")
        requirements = self._read(root / "requirements.txt")
        all_text = "\n".join(
            self._read(root / path)
            for path in paths[:300]
            if Path(path).suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".toml", ".txt", ".json"}
        )

        frameworks: list[str] = []
        language = "unknown"
        if any(path.endswith(".py") for path in paths) or "pyproject.toml" in root_files:
            language = "python"
        if any(path.endswith((".ts", ".tsx")) for path in paths):
            language = "typescript"
        elif any(path.endswith((".js", ".jsx")) for path in paths) and language == "unknown":
            language = "javascript"

        deps = json.dumps(package_json.get("dependencies", {})) + json.dumps(package_json.get("devDependencies", {}))
        python_deps = f"{pyproject}\n{requirements}\n{all_text}".lower()
        if "fastapi" in python_deps:
            frameworks.append("FastAPI")
        if "flask" in python_deps:
            frameworks.append("Flask")
        if "django" in python_deps or "manage.py" in root_files:
            frameworks.append("Django")
        if "next" in deps or "next.config.js" in root_files or "next.config.mjs" in root_files:
            frameworks.append("Next.js")
        if "react" in deps:
            frameworks.append("React")
        if "vite" in deps or "vite.config.ts" in root_files or "vite.config.js" in root_files:
            frameworks.append("Vite")

        package_managers = []
        if "package-lock.json" in root_files:
            package_managers.append("npm")
        if "pnpm-lock.yaml" in root_files:
            package_managers.append("pnpm")
        if "yarn.lock" in root_files:
            package_managers.append("yarn")
        if "pyproject.toml" in root_files:
            package_managers.append("pip/pyproject")
        if "requirements.txt" in root_files:
            package_managers.append("pip")

        databases = []
        if "postgres" in all_text.lower() or "psycopg" in python_deps or "pg" in deps:
            databases.append("PostgreSQL")
        if "sqlite" in all_text.lower() or "sqlite3" in python_deps:
            databases.append("SQLite")

        infrastructure = []
        if "Dockerfile" in root_files:
            infrastructure.append("Docker")
        if "docker-compose.yml" in root_files or "docker-compose.yaml" in root_files:
            infrastructure.append("Docker Compose")

        test_commands = self._test_commands(package_json, frameworks, language)
        build_commands = self._build_commands(package_json, frameworks, package_managers)
        return FrameworkProfile(
            language=language,
            frameworks=sorted(dict.fromkeys(frameworks)),
            package_managers=package_managers,
            databases=databases,
            infrastructure=infrastructure,
            test_commands=test_commands,
            build_commands=build_commands,
            conventions=self._conventions(frameworks),
        )

    def profile_for_objective(self, objective: str) -> FrameworkProfile:
        lowered = objective.lower()
        if any(term in lowered for term in ("website", "landing", "dashboard", "crm", "saas")):
            return FrameworkProfile(
                language="typescript",
                frameworks=["React", "Vite"],
                package_managers=["npm"],
                test_commands=["npm test"],
                build_commands=["npm run build"],
                conventions={"source": "src", "entrypoint": "index.html", "tests": "tests"},
            )
        if any(term in lowered for term in ("api", "oauth", "auth", "stripe", "fastapi")):
            return FrameworkProfile(
                language="python",
                frameworks=["FastAPI"],
                package_managers=["pip/pyproject"],
                databases=["SQLite"] if "sqlite" in lowered else [],
                test_commands=["python -m pytest -q"],
                build_commands=["python -m compileall app tests"],
                conventions={"source": "app", "entrypoint": "app/main.py", "tests": "tests"},
            )
        return FrameworkProfile(
            language="python",
            frameworks=[],
            package_managers=["pip/pyproject"],
            test_commands=["python -m pytest -q"],
            build_commands=["python -m compileall ."],
            conventions={"source": ".", "tests": "tests"},
        )

    @staticmethod
    def _package_json(root: Path) -> dict[str, Any]:
        path = root / "package.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        except OSError:
            return ""

    @staticmethod
    def _test_commands(package_json: dict[str, Any], frameworks: list[str], language: str) -> list[str]:
        scripts = package_json.get("scripts", {}) if isinstance(package_json.get("scripts"), dict) else {}
        commands: list[str] = []
        if "test" in scripts:
            commands.append("npm test")
        if language == "python":
            commands.append("python -m pytest -q")
        return commands

    @staticmethod
    def _build_commands(package_json: dict[str, Any], frameworks: list[str], package_managers: list[str]) -> list[str]:
        scripts = package_json.get("scripts", {}) if isinstance(package_json.get("scripts"), dict) else {}
        commands: list[str] = []
        if "build" in scripts:
            commands.append("npm run build")
        if "pip/pyproject" in package_managers:
            commands.append("python -m compileall .")
        return commands

    @staticmethod
    def _conventions(frameworks: list[str]) -> dict[str, str]:
        if "FastAPI" in frameworks:
            return {"source": "app", "entrypoint": "app/main.py", "tests": "tests"}
        if "Django" in frameworks:
            return {"source": "project", "entrypoint": "manage.py", "tests": "tests"}
        if "Next.js" in frameworks:
            return {"source": "app", "entrypoint": "app/page.tsx", "tests": "tests"}
        if "Vite" in frameworks or "React" in frameworks:
            return {"source": "src", "entrypoint": "index.html", "tests": "tests"}
        return {}
