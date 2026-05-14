from __future__ import annotations

import pytest

from backend.llm.router import ModelRole
from backend.runtime.lifecycle import (
    CognitionLifecycleManager,
    RuntimeEndpoint,
    RuntimeState,
)


@pytest.mark.asyncio
async def test_register_runtime() -> None:
    manager = CognitionLifecycleManager()
    runtime = await manager.register_runtime(
        runtime_id="coder-primary",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8000", model_name="deepseek"),
    )

    assert runtime.runtime_id == "coder-primary"
    assert runtime.state is RuntimeState.INACTIVE


@pytest.mark.asyncio
async def test_activate_runtime() -> None:
    manager = CognitionLifecycleManager()
    await manager.register_runtime(
        runtime_id="coder-primary",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8000", model_name="deepseek"),
    )

    runtime = await manager.activate_runtime("coder-primary", owner="runtime-manager")

    assert runtime.state is RuntimeState.ACTIVE
    assert runtime.owner == "runtime-manager"
    assert runtime.activation_timestamp is not None


@pytest.mark.asyncio
async def test_idempotent_activation() -> None:
    manager = CognitionLifecycleManager()
    await manager.register_runtime(
        runtime_id="judge",
        role=ModelRole.JUDGE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8001", model_name="judge-model"),
    )

    first = await manager.activate_runtime("judge", owner="judge-owner")
    second = await manager.activate_runtime("judge", owner="judge-owner")

    assert first is second
    assert second.state is RuntimeState.ACTIVE


@pytest.mark.asyncio
async def test_swap_behavior() -> None:
    manager = CognitionLifecycleManager()
    await manager.register_runtime(
        runtime_id="coder-a",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8000", model_name="coder-a"),
    )
    await manager.register_runtime(
        runtime_id="coder-b",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8001", model_name="coder-b"),
    )
    await manager.activate_runtime("coder-a", owner="owner-a")

    swapped = await manager.swap_runtime("coder-a", "coder-b", owner="owner-b")

    assert swapped.runtime_id == "coder-b"
    assert swapped.state is RuntimeState.ACTIVE
    assert swapped.owner == "owner-b"
    assert await manager.runtime_health("coder-a") is RuntimeState.INACTIVE


@pytest.mark.asyncio
async def test_endpoint_routing_by_role() -> None:
    manager = CognitionLifecycleManager()
    await manager.register_runtime(
        runtime_id="judge",
        role=ModelRole.JUDGE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:9000", model_name="judge-model"),
    )

    endpoint = await manager.inference_endpoint(role=ModelRole.JUDGE)

    assert endpoint is not None
    assert endpoint.base_url == "http://localhost:9000"


@pytest.mark.asyncio
async def test_health_transitions() -> None:
    manager = CognitionLifecycleManager()
    await manager.register_runtime(
        runtime_id="retry",
        role=ModelRole.RETRY_ENGINE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:9001", model_name="retry-model"),
        state=RuntimeState.LOADING,
    )

    assert await manager.runtime_health("retry") is RuntimeState.LOADING
    await manager.activate_runtime("retry")
    assert await manager.runtime_health("retry") is RuntimeState.ACTIVE
    await manager.deactivate_runtime("retry")
    assert await manager.runtime_health("retry") is RuntimeState.INACTIVE


@pytest.mark.asyncio
async def test_invalid_runtime_access() -> None:
    manager = CognitionLifecycleManager()

    with pytest.raises(KeyError, match="Unknown runtime_id"):
        await manager.activate_runtime("missing")


@pytest.mark.asyncio
async def test_active_runtime_ownership() -> None:
    manager = CognitionLifecycleManager()
    await manager.register_runtime(
        runtime_id="coder",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8000", model_name="coder"),
    )
    await manager.activate_runtime("coder", owner="owner-1")

    active = await manager.active_runtime()

    assert active is not None
    assert active.owner == "owner-1"


@pytest.mark.asyncio
async def test_sequential_orchestration_guarantees() -> None:
    manager = CognitionLifecycleManager()
    await manager.register_runtime(
        runtime_id="coder",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8000", model_name="coder"),
    )
    await manager.register_runtime(
        runtime_id="judge",
        role=ModelRole.JUDGE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8001", model_name="judge"),
    )

    await manager.activate_runtime("coder", owner="owner-coder")
    await manager.activate_runtime("judge", owner="owner-judge")

    coder_state = await manager.runtime_health("coder")
    judge_state = await manager.runtime_health("judge")
    active = await manager.active_runtime()

    assert coder_state is RuntimeState.INACTIVE
    assert judge_state is RuntimeState.ACTIVE
    assert active is not None
    assert active.runtime_id == "judge"
