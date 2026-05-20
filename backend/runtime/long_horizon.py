from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.runtime.context_compressor import CompressedRepositoryContext
from backend.runtime.task_planner import TaskPlan


@dataclass(slots=True)
class ExecutionGraphStep:
    step_id: str
    task_id: str
    kind: str
    title: str
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "title": self.title,
            "dependencies": self.dependencies,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class LongHorizonExecutionGraph:
    objective: str
    steps: list[ExecutionGraphStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "steps": [step.to_dict() for step in self.steps],
            "completed": [step.step_id for step in self.steps if step.status == "completed"],
            "blocked": [step.step_id for step in self.steps if step.status == "blocked"],
            "remaining": [step.step_id for step in self.steps if step.status == "pending"],
        }

    @classmethod
    def from_task_plan(cls, plan: TaskPlan) -> LongHorizonExecutionGraph:
        steps: list[ExecutionGraphStep] = []
        previous_task_final: str | None = None
        for task in plan.tasks:
            dependencies = list(task.dependencies)
            if previous_task_final and not dependencies:
                dependencies = [previous_task_final]
            patch_id = f"{task.task_id}:patch"
            test_id = f"{task.task_id}:tests"
            repair_id = f"{task.task_id}:repair"
            steps.extend(
                [
                    ExecutionGraphStep(
                        step_id=f"{task.task_id}:task",
                        task_id=task.task_id,
                        kind="task",
                        title=task.goal,
                        dependencies=dependencies,
                        metadata={"affected_files": task.affected_files},
                    ),
                    ExecutionGraphStep(
                        step_id=patch_id,
                        task_id=task.task_id,
                        kind="patch",
                        title=f"Patch for {task.goal}",
                        dependencies=[f"{task.task_id}:task"],
                    ),
                    ExecutionGraphStep(
                        step_id=test_id,
                        task_id=task.task_id,
                        kind="tests",
                        title=f"Validate {task.goal}",
                        dependencies=[patch_id],
                        metadata={"validation_strategy": task.validation_strategy},
                    ),
                    ExecutionGraphStep(
                        step_id=repair_id,
                        task_id=task.task_id,
                        kind="repair",
                        title=f"Repair {task.goal} if tests fail",
                        dependencies=[test_id],
                    ),
                ]
            )
            previous_task_final = repair_id
        return cls(objective=plan.objective, steps=steps)

    def mark_completed_by_kind(self, kind: str) -> None:
        for step in self.steps:
            if step.kind == kind:
                step.status = "completed"


@dataclass(slots=True)
class LongHorizonPreparation:
    task_plan: TaskPlan
    execution_graph: LongHorizonExecutionGraph
    compressed_context: CompressedRepositoryContext
    memory: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_plan": self.task_plan.to_dict(),
            "execution_graph": self.execution_graph.to_dict(),
            "compressed_context": self.compressed_context.to_dict(),
            "architecture_memory": self.memory,
        }


@dataclass(slots=True)
class ObjectiveMemoryRecord:
    objective: str
    repository_path: str
    plan: dict[str, Any]
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    outcome: str = "unknown"
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "repository_path": self.repository_path,
            "plan": self.plan,
            "artifacts": self.artifacts,
            "repairs": self.repairs,
            "failures": self.failures,
            "outcome": self.outcome,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveMemoryRecord:
        return cls(
            objective=str(data["objective"]),
            repository_path=str(data["repository_path"]),
            plan=dict(data.get("plan") or {}),
            artifacts=list(data.get("artifacts") or []),
            repairs=list(data.get("repairs") or []),
            failures=[str(item) for item in data.get("failures", [])],
            outcome=str(data.get("outcome", "unknown")),
            updated_at=str(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
        )


class ObjectiveMemory:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or ".forge/objective_memory.json").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: ObjectiveMemoryRecord) -> ObjectiveMemoryRecord:
        records = [
            item
            for item in self.list_records()
            if not (item.repository_path == record.repository_path and item.objective == record.objective)
        ]
        record.updated_at = datetime.now(timezone.utc).isoformat()
        records.append(record)
        self._save(records)
        return record

    def list_records(self, repository_path: str | None = None) -> list[ObjectiveMemoryRecord]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        records = [ObjectiveMemoryRecord.from_dict(item) for item in data.get("objectives", [])]
        if repository_path:
            resolved = str(Path(repository_path).resolve())
            records = [record for record in records if record.repository_path == resolved]
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def _save(self, records: list[ObjectiveMemoryRecord]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps({"objectives": [record.to_dict() for record in records]}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
