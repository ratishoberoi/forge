from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.runtime.architecture_memory import ArchitectureMemoryRecord, dependency_closure


@dataclass(slots=True)
class PlannedSubtask:
    subtask_id: str
    goal: str
    dependencies: list[str] = field(default_factory=list)
    validation_strategy: list[str] = field(default_factory=list)
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "goal": self.goal,
            "dependencies": self.dependencies,
            "validation_strategy": self.validation_strategy,
            "status": self.status,
        }


@dataclass(slots=True)
class PlannedEpic:
    epic_id: str
    title: str
    tasks: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "epic_id": self.epic_id,
            "title": self.title,
            "tasks": self.tasks,
            "dependencies": self.dependencies,
            "status": self.status,
        }


@dataclass(slots=True)
class PlannedTask:
    task_id: str
    goal: str
    affected_files: list[str]
    dependencies: list[str]
    validation_strategy: list[str]
    epic_id: str | None = None
    subtasks: list[PlannedSubtask] = field(default_factory=list)
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "affected_files": self.affected_files,
            "dependencies": self.dependencies,
            "validation_strategy": self.validation_strategy,
            "epic_id": self.epic_id,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "status": self.status,
        }


@dataclass(slots=True)
class TaskPlan:
    objective: str
    epics: list[PlannedEpic] = field(default_factory=list)
    tasks: list[PlannedTask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "epics": [epic.to_dict() for epic in self.epics],
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
        epics = self._epics(objective)
        goals = self._goals(objective)
        tasks: list[PlannedTask] = []
        previous_id: str | None = None
        for index, goal in enumerate(goals, start=1):
            task_id = f"task-{index}"
            epic = epics[min(index - 1, len(epics) - 1)] if epics else None
            affected = self._affected_files(goal, files, preparation, memory)
            dependencies = [previous_id] if previous_id else []
            tasks.append(
                PlannedTask(
                    task_id=task_id,
                    goal=goal,
                    affected_files=affected,
                    dependencies=dependencies,
                    validation_strategy=validation,
                    epic_id=epic.epic_id if epic else None,
                    subtasks=self._subtasks(task_id, goal, validation),
                )
            )
            previous_id = task_id
        if epics:
            by_epic: dict[str, list[str]] = {}
            for task in tasks:
                if task.epic_id:
                    by_epic.setdefault(task.epic_id, []).append(task.task_id)
            for index, epic in enumerate(epics):
                epic.tasks = by_epic.get(epic.epic_id, [])
                if index > 0:
                    epic.dependencies = [epics[index - 1].epic_id]
        return TaskPlan(objective=objective, epics=epics, tasks=tasks)

    def _epics(self, objective: str) -> list[PlannedEpic]:
        lowered = objective.lower()
        large_terms = {
            "saas",
            "billing",
            "rbac",
            "dashboard",
            "oauth",
            "multi-tenant",
            "tenant",
            "migrate",
            "migration",
            "refactor",
            "crm",
            "admin",
        }
        if not any(term in lowered for term in large_terms):
            return []
        return [
            PlannedEpic("epic-1", "Architecture discovery and risk mapping"),
            PlannedEpic("epic-2", "Implementation across affected boundaries"),
            PlannedEpic("epic-3", "Validation, repair, and convergence"),
        ]

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

    @staticmethod
    def _subtasks(task_id: str, goal: str, validation: list[str]) -> list[PlannedSubtask]:
        return [
            PlannedSubtask(
                subtask_id=f"{task_id}.1",
                goal=f"Gather context for: {goal}",
                validation_strategy=[],
            ),
            PlannedSubtask(
                subtask_id=f"{task_id}.2",
                goal=f"Apply repository-safe changes for: {goal}",
                dependencies=[f"{task_id}.1"],
                validation_strategy=validation,
            ),
            PlannedSubtask(
                subtask_id=f"{task_id}.3",
                goal=f"Validate and repair: {goal}",
                dependencies=[f"{task_id}.2"],
                validation_strategy=validation,
            ),
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
