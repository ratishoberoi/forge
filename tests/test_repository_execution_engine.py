from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from backend.config.settings import Settings
from backend.runtime.execution_runner import ExecutionRunner
from backend.runtime.autonomous_courtroom import AutonomousCourtroom
from backend.runtime.autonomous_run import AutonomousRun
from backend.runtime.context_budget import ContextBudgetManager
from backend.runtime.repository_execution_engine import (
    ObjectiveType,
    RepositoryExecutionEngine,
    RepositoryExecutionError,
    classify_objective,
)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        repo_index_state_path=str(tmp_path / ".forge" / "index_state.json"),
        repo_incremental=False,
    )


def test_repository_scan_detects_python_pytest_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='calculator'\n[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))

    prep = asyncio.run(engine.prepare("Build calculator app"))

    assert prep.scan.primary_language == "python"
    assert "pytest" in prep.scan.test_frameworks
    assert "pip/pyproject" in prep.scan.package_managers
    assert "app.py" in prep.scan.entrypoints
    assert prep.scan.build_commands


def test_context_builder_gathers_relevant_files_without_full_repo_dump(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "auth.py").write_text("def login():\n    return True\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    for index in range(12):
        (tmp_path / f"irrelevant_{index}.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))

    prep = asyncio.run(engine.prepare("Improve calculator subtraction"))

    assert "calculator.py" in prep.context.relevant_files
    assert "tests/test_calculator.py" in prep.context.related_tests
    assert len(prep.context.file_summaries) <= engine.MAX_CONTEXT_FILES
    assert "auth.py" not in prep.context.file_summaries


def test_planning_stage_for_build_calculator_creates_source_and_tests(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='calculator'\n", encoding="utf-8")
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))

    prep = asyncio.run(engine.prepare("Build calculator app"))

    assert "calculator.py" in prep.plan.files_to_create
    assert "tests/test_calculator.py" in prep.plan.files_to_create
    assert "tests/test_calculator.py" in prep.plan.expected_tests
    assert prep.plan.steps
    assert prep.plan.objective_type == ObjectiveType.APPLICATION


def test_build_calculator_app_can_write_source_tests_and_pass(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='calculator'\n", encoding="utf-8")
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))
    prep = asyncio.run(engine.prepare("Build calculator app"))
    response = json.dumps(
        {
            "summary": "Build calculator module and tests",
            "files": {
                "calculator.py": (
                    "def add(a: int, b: int) -> int:\n"
                    "    return a + b\n\n"
                    "def subtract(a: int, b: int) -> int:\n"
                    "    return a - b\n"
                ),
                "tests/test_calculator.py": (
                    "from calculator import add, subtract\n\n"
                    "def test_add():\n"
                    "    assert add(2, 3) == 5\n\n"
                    "def test_subtract():\n"
                    "    assert subtract(5, 3) == 2\n"
                ),
            },
        }
    )

    result = engine.apply_primary_output(response_text=response, plan=prep.plan)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    tests = ExecutionRunner(timeout=30).run(
        command=["pytest", "-q"],
        cwd=str(tmp_path),
        env=env,
    )

    assert result.success is True
    assert (tmp_path / "calculator.py").exists()
    assert (tmp_path / "tests" / "test_calculator.py").exists()
    assert tests.succeeded, tests.stderr


def test_objective_classification_distinguishes_patch_feature_application() -> None:
    assert classify_objective("Fix authentication bug") == ObjectiveType.PATCH
    assert classify_objective("Add OAuth") == ObjectiveType.FEATURE
    assert classify_objective("Build Todo application") == ObjectiveType.APPLICATION
    assert classify_objective("Build CRM") == ObjectiveType.APPLICATION
    assert classify_objective("Create SaaS landing page") == ObjectiveType.APPLICATION
    assert classify_objective("Refactor authentication system") == ObjectiveType.REFACTOR
    assert classify_objective("Convert Flask to FastAPI") == ObjectiveType.MIGRATION


def test_fastapi_todo_application_plan_requires_real_application_structure(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='todo'\ndependencies=['fastapi','uvicorn']\n[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n",
        encoding="utf-8",
    )
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))

    prep = asyncio.run(engine.prepare("Build a complete FastAPI Todo application"))

    required = prep.plan.acceptance_contract.required_files
    assert prep.plan.objective_type == ObjectiveType.APPLICATION
    assert "app/models.py" in required
    assert "app/database.py" in required
    assert "app/schemas.py" in required
    assert "app/repository.py" in required
    assert "requirements.txt" in required
    assert "tests/test_todos.py" in prep.plan.expected_tests
    assert "/todos" in prep.plan.acceptance_contract.required_routes


def test_fastapi_todo_application_replaces_bootstrap_placeholders(tmp_path: Path) -> None:
    _write_bootstrap_fastapi_stub(tmp_path)
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))

    prep = asyncio.run(engine.prepare("Build a complete FastAPI Todo application"))

    assert prep.plan.objective_type == ObjectiveType.APPLICATION
    assert len(prep.plan.files_to_create) >= 5
    assert "app/main.py" in prep.plan.files_to_create
    assert "app/models.py" in prep.plan.files_to_create
    assert "app/database.py" in prep.plan.files_to_create
    assert "app/schemas.py" in prep.plan.files_to_create
    assert "app/repository.py" in prep.plan.files_to_create
    assert "tests/test_todos.py" in prep.plan.files_to_create
    assert set(prep.plan.files_to_modify) != {"app/main.py", "tests/test_app.py"}


def test_budgeted_todo_prompt_stays_under_model_context_limit(tmp_path: Path) -> None:
    _write_bootstrap_fastapi_stub(tmp_path)
    for index in range(20):
        (tmp_path / f"notes_{index}.md").write_text("Todo context\n" * 500, encoding="utf-8")
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))
    prep = asyncio.run(engine.prepare("Build a complete FastAPI Todo application"))
    telemetry: list[str] = []

    prompt = AutonomousRun._objective_with_repository_context(
        "Build a complete FastAPI Todo application",
        prep,
        telemetry_callback=telemetry.append,
    )
    coder_prompt = AutonomousCourtroom._coder_prompt(prompt)
    system_prompt = AutonomousCourtroom._system_prompt_for_role("PRIMARY_CODER")
    estimate = (
        ContextBudgetManager().estimate_tokens(coder_prompt)
        + ContextBudgetManager().estimate_tokens(system_prompt)
        + 300
    )

    assert estimate < 8192
    assert any(item.startswith("[TOKEN_ESTIMATE]") for item in telemetry)
    assert any(item.startswith("[CONTEXT_BUDGET]") for item in telemetry)
    assert "app/models.py" in prompt
    assert "app/database.py" in prompt
    assert "objective_summary" not in prompt


def test_fastapi_todo_rejects_objective_summary_only_output(tmp_path: Path) -> None:
    _write_bootstrap_fastapi_stub(tmp_path)
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))
    prep = asyncio.run(engine.prepare("Build a complete FastAPI Todo application"))
    response = json.dumps(
        {
            "summary": "Update summary only",
            "files": {
                "app/main.py": (
                    "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
                    "def objective_summary() -> str:\n"
                    "    return 'Build a complete FastAPI Todo application'\n"
                ),
                "README.md": "# Todo\n\nPlaceholder objective summary.\n",
            },
        }
    )

    with pytest.raises(RepositoryExecutionError, match="missing required files|template"):
        engine.apply_primary_output(response_text=response, plan=prep.plan)


def test_fastapi_todo_application_writes_multifile_crud_app(tmp_path: Path) -> None:
    _write_bootstrap_fastapi_stub(tmp_path)
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))
    prep = asyncio.run(engine.prepare("Build a complete FastAPI Todo application"))
    response = json.dumps(_todo_primary_output())

    result = engine.apply_primary_output(response_text=response, plan=prep.plan)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    tests = ExecutionRunner(timeout=30).run(
        command=["pytest", "-q"],
        cwd=str(tmp_path),
        env=env,
    )

    assert result.success is True
    assert (tmp_path / "app" / "models.py").exists()
    assert (tmp_path / "app" / "database.py").exists()
    assert (tmp_path / "app" / "schemas.py").exists()
    assert (tmp_path / "app" / "repository.py").exists()
    assert (tmp_path / "tests" / "test_todos.py").exists()
    assert (tmp_path / "requirements.txt").exists()
    assert (tmp_path / "Dockerfile").exists()
    assert "/todos/{todo_id}" in (tmp_path / "app" / "main.py").read_text(encoding="utf-8")
    assert tests.succeeded, tests.stderr


def test_existing_fastapi_repo_adds_feature_to_loaded_repository(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    tests_dir = tmp_path / "tests"
    app_dir.mkdir()
    tests_dir.mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='api'\ndependencies=['fastapi']\n[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\ndef root():\n    return {'ok': True}\n",
        encoding="utf-8",
    )
    (tests_dir / "test_app.py").write_text(
        "from fastapi.testclient import TestClient\nfrom app.main import app\n\n"
        "def test_root():\n    assert TestClient(app).get('/').json() == {'ok': True}\n",
        encoding="utf-8",
    )
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))
    prep = asyncio.run(engine.prepare("Add a health endpoint"))

    response = json.dumps(
        {
            "summary": "Add health endpoint",
            "files": {
                "app/main.py": (
                    "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
                    "@app.get('/')\ndef root():\n    return {'ok': True}\n\n"
                    "@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n"
                ),
                "tests/test_app.py": (
                    "from fastapi.testclient import TestClient\nfrom app.main import app\n\n"
                    "def test_root():\n    assert TestClient(app).get('/').json() == {'ok': True}\n\n"
                    "def test_health():\n    assert TestClient(app).get('/health').json() == {'status': 'ok'}\n"
                ),
            },
        }
    )
    result = engine.apply_primary_output(response_text=response, plan=prep.plan)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    tests = ExecutionRunner(timeout=30).run(
        command=["pytest", "-q"],
        cwd=str(tmp_path),
        env=env,
    )

    assert "FastAPI" in prep.scan.frameworks
    assert "app/main.py" in prep.context.relevant_files
    assert result.success is True
    assert tests.succeeded, tests.stderr


def test_existing_react_repo_adds_page_to_selected_repository(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (tmp_path / "package.json").write_text(
        '{"scripts":{"test":"vitest --run","build":"vite build"},"dependencies":{"@vitejs/plugin-react":"latest","vite":"latest","react":"latest","react-dom":"latest"}}',
        encoding="utf-8",
    )
    (src / "App.tsx").write_text(
        "export function App() {\n  return <main>Home</main>;\n}\n",
        encoding="utf-8",
    )
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))
    prep = asyncio.run(engine.prepare("Add a settings page"))

    response = json.dumps(
        {
            "summary": "Add settings page",
            "files": {
                "src/App.tsx": (
                    "export function App() {\n"
                    "  return <main><h1>Home</h1><section aria-label=\"Settings\">Settings</section></main>;\n"
                    "}\n"
                ),
            },
        }
    )
    result = engine.apply_primary_output(response_text=response, plan=prep.plan)

    assert "React" in prep.scan.frameworks
    assert "src/App.tsx" in prep.context.relevant_files
    assert result.success is True
    assert "Settings" in (src / "App.tsx").read_text(encoding="utf-8")


def test_react_landing_page_plan_requires_application_files(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"scripts":{"test":"node --test tests/app.test.mjs","build":"node scripts/build.mjs"},"dependencies":{"react":"latest","react-dom":"latest","vite":"latest"}}',
        encoding="utf-8",
    )
    engine = RepositoryExecutionEngine(repo_root=str(tmp_path), settings=make_settings(tmp_path))

    prep = asyncio.run(engine.prepare("Build React Landing Page"))

    assert prep.plan.objective_type == ObjectiveType.APPLICATION
    assert "src/App.jsx" in prep.plan.acceptance_contract.required_files
    assert "tests/app.test.mjs" in prep.plan.expected_tests


def _write_bootstrap_fastapi_stub(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "[project]\nname='todo'\ndependencies=['fastapi','uvicorn','pytest']\n[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )
    (root / "app").mkdir()
    (root / "tests").mkdir()
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n\n"
        "def objective_summary() -> str:\n    return 'Build a complete FastAPI Todo application'\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_app.py").write_text(
        "from app.main import objective_summary\n\n"
        "def test_objective_summary():\n    assert objective_summary()\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Stub\n\nBootstrap placeholder.\n", encoding="utf-8")
    (root / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")


def _todo_primary_output() -> dict[str, object]:
    return {
        "summary": "Complete FastAPI Todo CRUD application",
        "files": {
            "requirements.txt": "fastapi\nuvicorn\npytest\n",
            "pyproject.toml": (
                "[project]\nname='todo'\ndependencies=['fastapi','uvicorn','pytest']\n"
                "[tool.pytest.ini_options]\ntestpaths=['tests']\n"
            ),
            "app/__init__.py": "# Todo application package\n",
            "app/database.py": (
                "from app.repository import TodoRepository\n\n"
                "repository = TodoRepository()\n"
            ),
            "app/schemas.py": "from app.models import Todo, TodoCreate\n",
            "app/models.py": (
                "from pydantic import BaseModel\n\n"
                "class TodoCreate(BaseModel):\n    title: str\n    completed: bool = False\n\n"
                "class Todo(TodoCreate):\n    id: int\n"
            ),
            "app/repository.py": (
                "from app.models import Todo, TodoCreate\n\n"
                "class TodoRepository:\n"
                "    def __init__(self):\n        self._items: dict[int, Todo] = {}\n        self._next_id = 1\n\n"
                "    def list(self) -> list[Todo]:\n        return list(self._items.values())\n\n"
                "    def create(self, payload: TodoCreate) -> Todo:\n"
                "        item = Todo(id=self._next_id, title=payload.title, completed=payload.completed)\n"
                "        self._items[item.id] = item\n        self._next_id += 1\n        return item\n\n"
                "    def get(self, todo_id: int) -> Todo | None:\n        return self._items.get(todo_id)\n\n"
                "    def update(self, todo_id: int, payload: TodoCreate) -> Todo | None:\n"
                "        if todo_id not in self._items:\n            return None\n"
                "        item = Todo(id=todo_id, title=payload.title, completed=payload.completed)\n"
                "        self._items[todo_id] = item\n        return item\n\n"
                "    def delete(self, todo_id: int) -> bool:\n        return self._items.pop(todo_id, None) is not None\n"
            ),
            "app/main.py": (
                "from fastapi import FastAPI, HTTPException\n"
                "from app.models import Todo, TodoCreate\n"
                "from app.database import repository\n\n"
                "app = FastAPI(title='Todo API')\nrepo = repository\n\n"
                "@app.get('/todos', response_model=list[Todo])\ndef list_todos():\n    return repo.list()\n\n"
                "@app.post('/todos', response_model=Todo, status_code=201)\ndef create_todo(payload: TodoCreate):\n    return repo.create(payload)\n\n"
                "@app.get('/todos/{todo_id}', response_model=Todo)\ndef get_todo(todo_id: int):\n"
                "    item = repo.get(todo_id)\n    if item is None:\n        raise HTTPException(status_code=404, detail='Todo not found')\n    return item\n\n"
                "@app.put('/todos/{todo_id}', response_model=Todo)\ndef update_todo(todo_id: int, payload: TodoCreate):\n"
                "    item = repo.update(todo_id, payload)\n    if item is None:\n        raise HTTPException(status_code=404, detail='Todo not found')\n    return item\n\n"
                "@app.delete('/todos/{todo_id}', status_code=204)\ndef delete_todo(todo_id: int):\n"
                "    if not repo.delete(todo_id):\n        raise HTTPException(status_code=404, detail='Todo not found')\n"
            ),
            "tests/test_todos.py": (
                "from fastapi.testclient import TestClient\nfrom app.main import app\n\n"
                "client = TestClient(app)\n\n"
                "def test_todo_crud():\n"
                "    created = client.post('/todos', json={'title': 'Ship Forge'}).json()\n"
                "    assert created['title'] == 'Ship Forge'\n"
                "    assert client.get('/todos').json()[0]['id'] == created['id']\n"
                "    updated = client.put(f\"/todos/{created['id']}\", json={'title': 'Ship Forge', 'completed': True}).json()\n"
                "    assert updated['completed'] is True\n"
                "    assert client.get(f\"/todos/{created['id']}\").json()['title'] == 'Ship Forge'\n"
                "    assert client.delete(f\"/todos/{created['id']}\").status_code == 204\n"
            ),
            "tests/test_app.py": (
                "from fastapi.testclient import TestClient\nfrom app.main import app\n\n"
                "def test_openapi_available():\n"
                "    assert TestClient(app).get('/openapi.json').status_code == 200\n"
            ),
            "README.md": "# Todo API\n\nFastAPI Todo CRUD application with create, read, update, and delete routes.\n",
            "Dockerfile": "FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nCMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\"]\n",
        },
    }
