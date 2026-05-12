"""Execution planning over repository intelligence context."""

from __future__ import annotations

from backend.config.settings import Settings, get_settings
from backend.repointel.context_builder import ContextBuilder
from backend.repointel.models import ExecutionPlan, PlanStep


class PlanningLayer:
    def __init__(
        self,
        context_builder: ContextBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved_settings = settings or get_settings()
        self._context_builder = context_builder or ContextBuilder(resolved_settings)

    async def plan(self, query: str) -> ExecutionPlan:
        context = await self._context_builder.build(query)
        impacted_files = context.related_files
        dependency_risks: list[str] = []
        steps: list[PlanStep] = []
        for file_path in impacted_files:
            neighbors = context.dependency_neighbors.get(file_path, [])
            if neighbors:
                dependency_risks.append(
                    f"{file_path} has {len(neighbors)} dependency neighbors that may require coordinated updates."
                )
            steps.append(
                PlanStep(
                    title=f"Inspect {file_path}",
                    description=f"Review retrieved symbols and dependencies in {file_path} before editing.",
                    file_paths=[file_path],
                    impact="medium" if neighbors else "low",
                )
            )
        if impacted_files:
            steps.append(
                PlanStep(
                    title="Validate impacted graph",
                    description="Re-run repository indexing and retrieval checks after changes to validate cross-file impact.",
                    file_paths=impacted_files,
                    impact="high",
                )
            )
        return ExecutionPlan(
            query=query,
            impacted_files=impacted_files,
            dependency_risks=dependency_risks,
            steps=steps,
        )
