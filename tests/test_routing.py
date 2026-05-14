from __future__ import annotations

import pytest

from backend.llm.router import ModelRole
from backend.repointel.models import CodeSymbol, ContextPackage, Language, RetrievalHit, SymbolKind
from backend.runtime.lifecycle import CognitionLifecycleManager, RuntimeEndpoint, RuntimeState
from backend.runtime.routing import CognitionRole, CognitionRouter


def make_context() -> ContextPackage:
    return ContextPackage(
        query="add retries to the worker",
        hits=[
            RetrievalHit(
                chunk_id="chunk-1",
                file_path="app/service.py",
                symbol_name="run_worker",
                language=Language.PYTHON,
                score=0.95,
                content="def run_worker():\n    return retry_loop()\n",
            ),
            RetrievalHit(
                chunk_id="chunk-2",
                file_path="app/retry.py",
                symbol_name="retry_loop",
                language=Language.PYTHON,
                score=0.91,
                content="def retry_loop():\n    return True\n",
            ),
        ],
        related_symbols=[
            CodeSymbol(
                id="sym-1",
                name="run_worker",
                kind=SymbolKind.FUNCTION,
                language=Language.PYTHON,
                file_path="app/service.py",
                start_line=1,
                end_line=2,
            )
        ],
        related_files=["app/service.py", "app/retry.py", "tests/test_retry.py"],
        dependency_neighbors={"app/service.py": ["app/retry.py", "app/config.py"]},
    )


@pytest.mark.asyncio
async def test_routing_selects_requested_role_and_runtime() -> None:
    lifecycle = CognitionLifecycleManager()
    await lifecycle.register_runtime(
        runtime_id="judge-runtime",
        role=ModelRole.JUDGE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8001", model_name="judge-model"),
        state=RuntimeState.ACTIVE,
    )
    router = CognitionRouter(lifecycle_manager=lifecycle)

    decision = await router.route_task(
        {"title": "compare candidate patches"},
        requested_role=CognitionRole.JUDGE,
        repository_context=make_context(),
        critiques=["candidate-a is larger than necessary"],
    )

    assert decision.role is CognitionRole.JUDGE
    assert decision.runtime_id == "judge-runtime"
    assert decision.endpoint is not None
    assert decision.runtime_role is ModelRole.JUDGE


def test_routing_role_selection_is_deterministic() -> None:
    router = CognitionRouter()

    assert router.select_role("judge these candidates") is CognitionRole.JUDGE
    assert router.select_role("repair the failing patch after pytest") is CognitionRole.RETRY_SPECIALIST
    assert router.select_role("architecture review for dependency impact") is CognitionRole.ARCHITECT
    assert router.select_role("synthesize repository findings") is CognitionRole.SYNTHESIZER
    assert router.select_role("implement the requested code change") is CognitionRole.PRIMARY_CODER


def test_budget_allocation_honors_role_specific_ceilings() -> None:
    router = CognitionRouter()

    primary = router.allocate_budget(CognitionRole.PRIMARY_CODER, requested_tokens=5000, context_token_estimate=2000)
    retry = router.allocate_budget(CognitionRole.RETRY_SPECIALIST, requested_tokens=5000, context_token_estimate=2000)

    assert primary.allocated_tokens > retry.allocated_tokens
    assert primary.role_ceiling_tokens > retry.role_ceiling_tokens
    assert primary.max_context_tokens >= primary.allocated_tokens


def test_context_shaping_is_role_specific() -> None:
    router = CognitionRouter()
    context = make_context()

    judge_context = router.shape_context(
        CognitionRole.JUDGE,
        task="judge candidate patches",
        repository_context=context,
        patch_summaries=["small diff"],
        execution_summaries=["pytest failed"],
        critiques=["too broad"],
        architecture_notes=["touches retry stack"],
    )
    architect_context = router.shape_context(
        CognitionRole.ARCHITECT,
        task="architecture review",
        repository_context=context,
        patch_summaries=["small diff", "extra diff"],
        execution_summaries=["pytest failed"],
        critiques=["too broad"],
        architecture_notes=["touches retry stack"],
    )

    assert len(judge_context.repository_snippets) <= 2
    assert architect_context.execution_summaries == ()
    assert architect_context.architecture_notes == ("touches retry stack",)


def test_oversized_context_sets_truncation_and_memory_hooks() -> None:
    router = CognitionRouter(memory_hooks=("persistent_memory", "vector_retrieval"))
    context = router.shape_context(
        CognitionRole.PRIMARY_CODER,
        task="implement feature",
        patch_summaries=["x" * 16000],
    )

    budget = router.allocate_budget(
        CognitionRole.PRIMARY_CODER,
        requested_tokens=3000,
        context_token_estimate=context.token_estimate,
    )

    assert budget.truncated is True
    assert budget.compression_hooks == ("persistent_memory", "vector_retrieval")


@pytest.mark.asyncio
async def test_route_task_produces_stable_decision_and_prompt_context() -> None:
    lifecycle = CognitionLifecycleManager()
    await lifecycle.register_runtime(
        runtime_id="coder-runtime",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8000", model_name="coder-model"),
        state=RuntimeState.INACTIVE,
    )
    router = CognitionRouter(lifecycle_manager=lifecycle)

    first = await router.route_task(
        {"summary": "implement worker retry behavior"},
        repository_context=make_context(),
        patch_summaries=["adjust retry backoff"],
        execution_summaries=["pytest passed"],
        critiques=["keep diff narrow"],
        memory_summaries=["previous retry strategy summary"],
        owner="task-1",
    )
    second = await router.route_task(
        {"summary": "implement worker retry behavior"},
        repository_context=make_context(),
        patch_summaries=["adjust retry backoff"],
        execution_summaries=["pytest passed"],
        critiques=["keep diff narrow"],
        memory_summaries=["previous retry strategy summary"],
        owner="task-1",
    )

    assert first.role is CognitionRole.PRIMARY_CODER
    assert first.runtime_id == "coder-runtime"
    assert first.prompt_context == second.prompt_context
    assert "Task Summary:" in first.prompt_context
    assert "Patch Summary:" in first.prompt_context
    assert first.budget.allocated_tokens <= first.budget.max_context_tokens
