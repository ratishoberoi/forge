from __future__ import annotations

import pytest

from backend.llm.router import ModelRole
from backend.runtime.binding import RuntimeBinder
from backend.runtime.lifecycle import (
    CognitionLifecycleManager,
    RuntimeEndpoint,
    RuntimeState,
)
from backend.runtime.routing import CognitionRole, CognitionRouter


@pytest.mark.asyncio
async def test_role_binding_and_lookup() -> None:
    lifecycle = CognitionLifecycleManager()
    router = CognitionRouter(lifecycle_manager=lifecycle)
    binder = RuntimeBinder(lifecycle_manager=lifecycle, router=router)

    binding = binder.bind_role(CognitionRole.PRIMARY_CODER, "coder-a", fallback_runtime_ids=("coder-b",))

    assert binding.runtime_id == "coder-a"
    assert binder.binding_for_role(CognitionRole.PRIMARY_CODER) is binding


@pytest.mark.asyncio
async def test_deterministic_dispatch_uses_bound_runtime() -> None:
    lifecycle = CognitionLifecycleManager()
    await lifecycle.register_runtime(
        runtime_id="coder-a",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8000", model_name="coder-a"),
        state=RuntimeState.INACTIVE,
    )
    await lifecycle.register_runtime(
        runtime_id="coder-b",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8001", model_name="coder-b"),
        state=RuntimeState.ACTIVE,
    )
    binder = RuntimeBinder(lifecycle_manager=lifecycle)
    binder.bind_role(CognitionRole.PRIMARY_CODER, "coder-a", fallback_runtime_ids=("coder-b",))

    decision = await binder.dispatch_runtime("implement the feature", owner="task-1")

    assert decision.role is CognitionRole.PRIMARY_CODER
    assert decision.runtime_id == "coder-a"
    assert decision.used_fallback is False
    assert decision.endpoint is not None
    assert decision.endpoint.base_url == "http://localhost:8000"


@pytest.mark.asyncio
async def test_runtime_swapping_updates_binding_and_activation() -> None:
    lifecycle = CognitionLifecycleManager()
    await lifecycle.register_runtime(
        runtime_id="judge-a",
        role=ModelRole.JUDGE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:9000", model_name="judge-a"),
    )
    await lifecycle.register_runtime(
        runtime_id="judge-b",
        role=ModelRole.JUDGE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:9001", model_name="judge-b"),
    )
    binder = RuntimeBinder(lifecycle_manager=lifecycle)
    binder.bind_role(CognitionRole.JUDGE, "judge-a")
    await binder.activate_binding(CognitionRole.JUDGE, owner="judge-owner")

    decision = await binder.swap_binding(CognitionRole.JUDGE, "judge-b", owner="judge-owner")

    assert decision.runtime_id == "judge-b"
    assert binder.binding_for_role(CognitionRole.JUDGE).runtime_id == "judge-b"
    assert await lifecycle.runtime_health("judge-a") is RuntimeState.INACTIVE
    assert await lifecycle.runtime_health("judge-b") is RuntimeState.ACTIVE


@pytest.mark.asyncio
async def test_fallback_behavior_on_unavailable_runtime() -> None:
    lifecycle = CognitionLifecycleManager()
    await lifecycle.register_runtime(
        runtime_id="retry-primary",
        role=ModelRole.RETRY_ENGINE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:9100", model_name="retry-a"),
        state=RuntimeState.FAILED,
    )
    await lifecycle.register_runtime(
        runtime_id="retry-fallback",
        role=ModelRole.RETRY_ENGINE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:9101", model_name="retry-b"),
        state=RuntimeState.INACTIVE,
    )
    binder = RuntimeBinder(lifecycle_manager=lifecycle)
    binder.bind_role(CognitionRole.RETRY_SPECIALIST, "retry-primary", fallback_runtime_ids=("retry-fallback",))

    decision = await binder.dispatch_runtime(
        "repair patch after pytest failure",
        requested_role=CognitionRole.RETRY_SPECIALIST,
    )

    assert decision.runtime_id == "retry-fallback"
    assert decision.used_fallback is True
    assert decision.degraded is True


@pytest.mark.asyncio
async def test_unavailable_runtime_handling_without_fallback() -> None:
    lifecycle = CognitionLifecycleManager()
    await lifecycle.register_runtime(
        runtime_id="architect-a",
        role=ModelRole.ARCHITECTURE_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:9200", model_name="architect-a"),
        state=RuntimeState.UNHEALTHY,
    )
    binder = RuntimeBinder(lifecycle_manager=lifecycle)
    binder.bind_role(CognitionRole.ARCHITECT, "architect-a")

    decision = await binder.dispatch_runtime(
        "architecture review for dependency impact",
        requested_role=CognitionRole.ARCHITECT,
    )

    assert decision.runtime_id is None
    assert decision.degraded is True
    assert decision.health is None


@pytest.mark.asyncio
async def test_active_runtime_ownership_and_bounded_heavyweight_activation() -> None:
    lifecycle = CognitionLifecycleManager()
    await lifecycle.register_runtime(
        runtime_id="coder",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8000", model_name="coder"),
    )
    await lifecycle.register_runtime(
        runtime_id="judge",
        role=ModelRole.JUDGE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8001", model_name="judge"),
    )
    binder = RuntimeBinder(lifecycle_manager=lifecycle)
    binder.bind_role(CognitionRole.PRIMARY_CODER, "coder")
    binder.bind_role(CognitionRole.JUDGE, "judge")

    await binder.activate_binding(CognitionRole.PRIMARY_CODER, owner="coder-owner")
    decision = await binder.activate_binding(CognitionRole.JUDGE, owner="judge-owner")
    active = await binder.active_binding()

    assert decision.runtime_id == "judge"
    assert await lifecycle.runtime_health("coder") is RuntimeState.INACTIVE
    assert active is not None
    assert active.runtime_id == "judge"
    assert active.owner == "judge-owner"


@pytest.mark.asyncio
async def test_stable_binding_decisions() -> None:
    lifecycle = CognitionLifecycleManager()
    await lifecycle.register_runtime(
        runtime_id="synth",
        role=ModelRole.REPO_SYNTHESIZER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:9300", model_name="synth"),
        state=RuntimeState.INACTIVE,
    )
    binder = RuntimeBinder(lifecycle_manager=lifecycle)
    binder.bind_role(CognitionRole.SYNTHESIZER, "synth")

    first = await binder.dispatch_runtime(
        "synthesize repository findings",
        requested_role=CognitionRole.SYNTHESIZER,
        owner="task-9",
    )
    second = await binder.dispatch_runtime(
        "synthesize repository findings",
        requested_role=CognitionRole.SYNTHESIZER,
        owner="task-9",
    )

    assert first.runtime_id == second.runtime_id
    assert first.role is second.role
    assert first.used_fallback is second.used_fallback
