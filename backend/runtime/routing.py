"""Deterministic cognition routing and context allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from backend.config.settings import Settings, get_settings
from backend.llm.router import ModelRole
from backend.repointel.models import ContextPackage
from backend.runtime.context_budget import ContextBudgetManager, ContextChunk
from backend.runtime.lifecycle import CognitionLifecycleManager, RuntimeEndpoint

_APPROX_CHARS_PER_TOKEN = 4


class CognitionRole(StrEnum):
    PRIMARY_CODER = "primary_coder"
    JUDGE = "judge"
    RETRY_SPECIALIST = "retry_specialist"
    ARCHITECT = "architect"
    SYNTHESIZER = "synthesizer"


@dataclass(slots=True)
class ContextBudget:
    requested_tokens: int
    allocated_tokens: int
    max_context_tokens: int
    role_ceiling_tokens: int
    reserved_output_tokens: int
    truncated: bool
    compression_hooks: tuple[str, ...] = ()


@dataclass(slots=True)
class RoutedContext:
    role: CognitionRole
    summary: str
    repository_files: tuple[str, ...]
    repository_snippets: tuple[str, ...]
    patch_summaries: tuple[str, ...]
    execution_summaries: tuple[str, ...]
    critiques: tuple[str, ...]
    architecture_notes: tuple[str, ...]
    memory_hooks: tuple[str, ...]
    token_estimate: int
    truncated: bool = False


@dataclass(slots=True)
class RoutingDecision:
    role: CognitionRole
    runtime_id: str | None
    endpoint: RuntimeEndpoint | None
    budget: ContextBudget
    context: RoutedContext
    prompt_context: str
    ownership: str | None = None
    runtime_role: ModelRole | None = None


class CognitionRouter:
    """Routes tasks to cognition roles, runtimes, and bounded prompt context."""

    _ROLE_TO_MODEL_ROLE: dict[CognitionRole, ModelRole] = {
        CognitionRole.PRIMARY_CODER: ModelRole.PRIMARY_CODER,
        CognitionRole.JUDGE: ModelRole.JUDGE,
        CognitionRole.RETRY_SPECIALIST: ModelRole.RETRY_ENGINE,
        CognitionRole.ARCHITECT: ModelRole.ARCHITECTURE_CODER,
        CognitionRole.SYNTHESIZER: ModelRole.REPO_SYNTHESIZER,
    }

    _ROLE_CEILING_RATIO: dict[CognitionRole, float] = {
        CognitionRole.PRIMARY_CODER: 0.90,
        CognitionRole.JUDGE: 0.75,
        CognitionRole.RETRY_SPECIALIST: 0.60,
        CognitionRole.ARCHITECT: 0.85,
        CognitionRole.SYNTHESIZER: 0.70,
    }

    _ROLE_OUTPUT_RESERVE: dict[CognitionRole, int] = {
        CognitionRole.PRIMARY_CODER: 1024,
        CognitionRole.JUDGE: 768,
        CognitionRole.RETRY_SPECIALIST: 512,
        CognitionRole.ARCHITECT: 896,
        CognitionRole.SYNTHESIZER: 640,
    }

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        lifecycle_manager: CognitionLifecycleManager | None = None,
        memory_hooks: tuple[str, ...] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._lifecycle_manager = lifecycle_manager
        self._memory_hooks = tuple(memory_hooks or ("persistent_memory", "vector_retrieval", "compressed_summaries"))

    async def route_task(
        self,
        task: str | dict[str, object],
        *,
        requested_role: CognitionRole | str | None = None,
        repository_context: ContextPackage | None = None,
        patch_summaries: list[str] | tuple[str, ...] | None = None,
        execution_summaries: list[str] | tuple[str, ...] | None = None,
        critiques: list[str] | tuple[str, ...] | None = None,
        architecture_notes: list[str] | tuple[str, ...] | None = None,
        memory_summaries: list[str] | tuple[str, ...] | None = None,
        requested_tokens: int | None = None,
        owner: str | None = None,
    ) -> RoutingDecision:
        role = self.select_role(task, requested_role=requested_role)
        context = self.shape_context(
            role,
            task=task,
            repository_context=repository_context,
            patch_summaries=patch_summaries,
            execution_summaries=execution_summaries,
            critiques=critiques,
            architecture_notes=architecture_notes,
            memory_summaries=memory_summaries,
        )
        budget = self.allocate_budget(
            role,
            requested_tokens=requested_tokens,
            context_token_estimate=context.token_estimate,
        )
        prompt_context = self.prepare_prompt_context(context, budget)
        runtime_id, endpoint = await self.select_runtime(role, owner=owner)
        model_role = self._ROLE_TO_MODEL_ROLE[role]
        return RoutingDecision(
            role=role,
            runtime_id=runtime_id,
            endpoint=endpoint,
            budget=budget,
            context=context,
            prompt_context=prompt_context,
            ownership=owner,
            runtime_role=model_role,
        )

    def select_role(
        self,
        task: str | dict[str, object],
        *,
        requested_role: CognitionRole | str | None = None,
    ) -> CognitionRole:
        if requested_role is not None:
            return self._normalize_role(requested_role)

        task_text = self._task_text(task).lower()
        if any(keyword in task_text for keyword in ("judge", "score", "rank", "compare candidate", "critique patch")):
            return CognitionRole.JUDGE
        if (
            any(keyword in task_text for keyword in ("repair", "revise", "fix failure", "repair patch"))
            or (
                "retry" in task_text
                and any(keyword in task_text for keyword in ("failure", "pytest", "test failure", "execution"))
            )
        ):
            return CognitionRole.RETRY_SPECIALIST
        if any(keyword in task_text for keyword in ("architecture", "design", "dependency impact", "system change")):
            return CognitionRole.ARCHITECT
        if any(keyword in task_text for keyword in ("synthesize", "summarize repo", "aggregate")):
            return CognitionRole.SYNTHESIZER
        return CognitionRole.PRIMARY_CODER

    def allocate_budget(
        self,
        role: CognitionRole,
        *,
        requested_tokens: int | None = None,
        context_token_estimate: int = 0,
    ) -> ContextBudget:
        max_context_tokens = self._settings.runtime_context_token_budget
        role_ceiling_tokens = max(
            256,
            int(max_context_tokens * self._ROLE_CEILING_RATIO[role]),
        )
        reserved_output_tokens = min(self._ROLE_OUTPUT_RESERVE[role], role_ceiling_tokens // 2)
        usable_tokens = max(128, role_ceiling_tokens - reserved_output_tokens)
        requested = min(requested_tokens or usable_tokens, max_context_tokens)
        allocated = min(requested, usable_tokens)
        truncated = context_token_estimate > allocated
        compression_hooks = self._memory_hooks if truncated else ()
        return ContextBudget(
            requested_tokens=requested,
            allocated_tokens=allocated,
            max_context_tokens=max_context_tokens,
            role_ceiling_tokens=role_ceiling_tokens,
            reserved_output_tokens=reserved_output_tokens,
            truncated=truncated,
            compression_hooks=compression_hooks,
        )

    def shape_context(
        self,
        role: CognitionRole,
        *,
        task: str | dict[str, object],
        repository_context: ContextPackage | None = None,
        patch_summaries: list[str] | tuple[str, ...] | None = None,
        execution_summaries: list[str] | tuple[str, ...] | None = None,
        critiques: list[str] | tuple[str, ...] | None = None,
        architecture_notes: list[str] | tuple[str, ...] | None = None,
        memory_summaries: list[str] | tuple[str, ...] | None = None,
    ) -> RoutedContext:
        summary = self._task_text(task)
        repo_files = self._repository_files(repository_context, role)
        repo_snippets = self._repository_snippets(repository_context, role)
        patch_items = tuple(patch_summaries or ())
        execution_items = tuple(execution_summaries or ())
        critique_items = tuple(critiques or ())
        architecture_items = tuple(architecture_notes or ())
        memory_hooks = tuple(memory_summaries or ())

        if role is CognitionRole.JUDGE:
            repo_snippets = repo_snippets[:2]
        elif role is CognitionRole.RETRY_SPECIALIST:
            repo_files = repo_files[:4]
            architecture_items = ()
        elif role is CognitionRole.ARCHITECT:
            patch_items = patch_items[:2]
            execution_items = ()
        elif role is CognitionRole.SYNTHESIZER:
            execution_items = ()

        token_estimate = self._estimate_tokens(summary)
        token_estimate += sum(self._estimate_tokens(item) for item in repo_files)
        token_estimate += sum(self._estimate_tokens(item) for item in repo_snippets)
        token_estimate += sum(self._estimate_tokens(item) for item in patch_items)
        token_estimate += sum(self._estimate_tokens(item) for item in execution_items)
        token_estimate += sum(self._estimate_tokens(item) for item in critique_items)
        token_estimate += sum(self._estimate_tokens(item) for item in architecture_items)
        token_estimate += sum(self._estimate_tokens(item) for item in memory_hooks)

        return RoutedContext(
            role=role,
            summary=summary,
            repository_files=repo_files,
            repository_snippets=repo_snippets,
            patch_summaries=patch_items,
            execution_summaries=execution_items,
            critiques=critique_items,
            architecture_notes=architecture_items,
            memory_hooks=memory_hooks,
            token_estimate=token_estimate,
        )

    async def select_runtime(
        self,
        role: CognitionRole,
        *,
        owner: str | None = None,
    ) -> tuple[str | None, RuntimeEndpoint | None]:
        del owner
        if self._lifecycle_manager is None:
            return None, None
        runtime = await self._lifecycle_manager.runtime_for_role(self._ROLE_TO_MODEL_ROLE[role])
        if runtime is None:
            return None, None
        return runtime.runtime_id, runtime.endpoint

    def prepare_prompt_context(self, context: RoutedContext, budget: ContextBudget) -> str:
        budget_manager = ContextBudgetManager(max_tokens=budget.allocated_tokens)
        chunks: list[ContextChunk] = [
            ContextChunk(content=f"Task Summary:\n{context.summary}", priority=100),
        ]
        chunks.extend(ContextChunk(content=f"Repository File:\n{item}", priority=95) for item in context.repository_files)
        chunks.extend(
            ContextChunk(content=f"Repository Snippet:\n{item}", priority=90) for item in context.repository_snippets
        )
        chunks.extend(ContextChunk(content=f"Patch Summary:\n{item}", priority=85) for item in context.patch_summaries)
        chunks.extend(
            ContextChunk(content=f"Execution Evidence:\n{item}", priority=80) for item in context.execution_summaries
        )
        chunks.extend(ContextChunk(content=f"Critique:\n{item}", priority=75) for item in context.critiques)
        chunks.extend(
            ContextChunk(content=f"Architecture Note:\n{item}", priority=70) for item in context.architecture_notes
        )
        chunks.extend(ContextChunk(content=f"Memory Hook:\n{item}", priority=65) for item in context.memory_hooks)
        prompt_context = budget_manager.build_context(chunks)
        context.truncated = budget.truncated
        return prompt_context

    @staticmethod
    def _normalize_role(role: CognitionRole | str) -> CognitionRole:
        if isinstance(role, CognitionRole):
            return role
        return CognitionRole(role)

    @staticmethod
    def _task_text(task: str | dict[str, object]) -> str:
        if isinstance(task, str):
            return task
        for key in ("summary", "title", "query", "description", "task"):
            value = task.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return str(task)

    @staticmethod
    def _repository_files(
        repository_context: ContextPackage | None,
        role: CognitionRole,
    ) -> tuple[str, ...]:
        if repository_context is None:
            return ()
        files = sorted(dict.fromkeys(repository_context.related_files))
        if role is CognitionRole.JUDGE:
            return tuple(files[:3])
        if role is CognitionRole.RETRY_SPECIALIST:
            return tuple(files[:5])
        return tuple(files[:6])

    @staticmethod
    def _repository_snippets(
        repository_context: ContextPackage | None,
        role: CognitionRole,
    ) -> tuple[str, ...]:
        if repository_context is None:
            return ()
        snippets = [hit.content for hit in repository_context.hits]
        if role is CognitionRole.ARCHITECT:
            snippets.extend(
                f"{path}: {', '.join(sorted(neighbors))}"
                for path, neighbors in sorted(repository_context.dependency_neighbors.items())
            )
        limit = 4 if role is CognitionRole.PRIMARY_CODER else 3
        return tuple(snippets[:limit])

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // _APPROX_CHARS_PER_TOKEN) if text else 0
