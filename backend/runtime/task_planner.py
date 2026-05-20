from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.runtime.architecture_memory import ArchitectureMemoryRecord, dependency_closure


@dataclass(slots=True)
class PlannedTask:
    task_id: str
    goal: str
    affected_files: list[str]
    dependencies: list[str]
    validation_strategy: list[str]
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "affected_files": self.affected_files,
            "dependencies": self.dependencies,
            "validation_strategy": self.validation_strategy,
            "status": self.status,
        }


@dataclass(slots=True)
class TaskPlan:
    objective: str
    tasks: list[PlannedTask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "tasks": [task.to_dict() for task in self.tasks],
        }


class TaskPlanner:
    """Deterministic repository-scale task decomposition."""

    def plan(
        self,
        *,
        objective: str,
        preparation: Any,
        memory: ArchitectureMemoryRecord | None = None,
    ) -> TaskPlan:
        files = _dedupe(
            preparation.plan.files_to_modify
            + preparation.plan.files_to_create
            + preparation.context.relevant_files
        )
        if memory:
            files = dependency_closure(memory.dependency_graph, files, max_depth=2, max_files=50)
        validation = preparation.scan.test_commands or ["pytest -q"]
        goals = self._goals(objective)
        tasks: list[PlannedTask] = []
        previous_id: str | None = None
        for index, goal in enumerate(goals, start=1):
            task_id = f"task-{index}"
            affected = self._affected_files(goal, files, preparation, memory)
            dependencies = [previous_id] if previous_id else []
            tasks.append(
                PlannedTask(
                    task_id=task_id,
                    goal=goal,
                    affected_files=affected,
                    dependencies=dependencies,
                    validation_strategy=validation,
                )
            )
            previous_id = task_id
        return TaskPlan(objective=objective, tasks=tasks)

    def _goals(self, objective: str) -> list[str]:
        lowered = objective.lower()
        if "flask" in lowered and "fastapi" in lowered:
            return [
                "Inventory Flask entrypoints, routes, middleware, and tests.",
                "Introduce FastAPI application structure while preserving behavior.",
                "Migrate route handlers and dependency wiring.",
                "Update tests and execution commands for FastAPI.",
            ]
        if "oauth" in lowered or "login" in lowered or "auth" in lowered:
            return [
                "Map authentication entrypoints, user model, configuration, and tests.",
                "Implement provider configuration, callback handling, and session integration.",
                "Update protected routes and tests for the authentication flow.",
                "Validate security risks, failure handling, and regression coverage.",
            ]
        if "dashboard" in lowered:
            return [
                "Identify frontend shell, routing, data sources, and tests.",
                "Implement dashboard layout and state/data integration.",
                "Add or update tests for dashboard behavior.",
            ]
        if "multi-tenant" in lowered or "tenant" in lowered:
            return [
                "Map tenant-sensitive data models, auth boundaries, and query paths.",
                "Introduce tenant identity propagation and storage constraints.",
                "Update service logic and tests for tenant isolation.",
                "Validate migration, access-control, and regression risks.",
            ]
        return [
            "Analyze affected architecture and plan file-level changes.",
            f"Implement objective: {objective}",
            "Update tests and validation for changed behavior.",
        ]

    def _affected_files(
        self,
        goal: str,
        files: list[str],
        preparation: Any,
        memory: ArchitectureMemoryRecord | None,
    ) -> list[str]:
        terms = set(re.findall(r"[A-Za-z0-9]+", goal.lower()))
        scored: list[tuple[int, str]] = []
        important = set(memory.important_modules if memory else [])
        for path in files:
            score = len(terms & set(re.findall(r"[A-Za-z0-9]+", path.lower()))) * 5
            if path in preparation.context.related_tests:
                score += 2
            if path in important:
                score += 3
            scored.append((score, path))
        selected = [path for _, path in sorted(scored, key=lambda item: (-item[0], item[1]))[:12]]
        return selected or files[:12]


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
