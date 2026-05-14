from __future__ import annotations

import pytest

from backend.runtime.binding import RuntimeBinder
from backend.runtime.cognition import CognitionResponse as AdapterResponse
from backend.runtime.fabric import (
    CognitionExecutionFabric,
    CognitionRequest,
    InferencePolicy,
)
from backend.runtime.lifecycle import (
    CognitionLifecycleManager,
    RuntimeEndpoint,
    RuntimeState,
)
from backend.runtime.routing import CognitionRole, CognitionRouter
from backend.llm.router import ModelRole


class FakeCognitionAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_role: ModelRole,
        temperature: float,
        max_tokens: int,
        agent_id: str | None = None,
    ) -> AdapterResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model_role": model_role,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "agent_id": agent_id,
            }
        )
        return AdapterResponse(
            content=f"{model_role.value}:ok",
            model=f"{model_role.value}-model",
            prompt_tokens=64,
            completion_tokens=32,
            finish_reason="stop",
        )


async def make_fabric() -> tuple[CognitionExecutionFabric, CognitionLifecycleManager, FakeCognitionAdapter]:
    lifecycle = CognitionLifecycleManager()
    await lifecycle.register_runtime(
        runtime_id="coder-runtime",
        role=ModelRole.PRIMARY_CODER,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8000", model_name="coder-model"),
        state=RuntimeState.INACTIVE,
    )
    await lifecycle.register_runtime(
        runtime_id="judge-runtime",
        role=ModelRole.JUDGE,
        endpoint=RuntimeEndpoint(base_url="http://localhost:8001", model_name="judge-model"),
        state=RuntimeState.INACTIVE,
    )
    router = CognitionRouter(lifecycle_manager=lifecycle)
    binder = RuntimeBinder(lifecycle_manager=lifecycle, router=router)
    binder.bind_role(CognitionRole.PRIMARY_CODER, "coder-runtime")
    binder.bind_role(CognitionRole.JUDGE, "judge-runtime")
    adapter = FakeCognitionAdapter()
    fabric = CognitionExecutionFabric(
        binder=binder,
        router=router,
        lifecycle_manager=lifecycle,
        cognition_adapter=adapter,
    )
    return fabric, lifecycle, adapter


@pytest.mark.asyncio
async def test_role_execution_and_runtime_dispatch() -> None:
    fabric, lifecycle, adapter = await make_fabric()

    response = await fabric.execute_for_role(
        CognitionRole.PRIMARY_CODER,
        "implement the feature",
        owner="task-1",
        agent_id="agent-1",
    )

    assert response.role is CognitionRole.PRIMARY_CODER
    assert response.runtime_id == "coder-runtime"
    assert response.model == "primary_coder-model"
    assert adapter.calls[0]["model_role"] is ModelRole.PRIMARY_CODER
    assert await lifecycle.runtime_health("coder-runtime") is RuntimeState.ACTIVE


@pytest.mark.asyncio
async def test_policy_application_supports_role_defaults_and_overrides() -> None:
    fabric, _, adapter = await make_fabric()

    response = await fabric.execute(
        CognitionRequest(
            task="judge these candidates",
            requested_role=CognitionRole.JUDGE,
            owner="judge-owner",
            policy_override=InferencePolicy(max_tokens=512, critique_intensity="very_high"),
        )
    )

    assert response.policy.temperature == 0.0
    assert response.policy.max_tokens == 512
    assert response.policy.critique_intensity == "very_high"
    assert adapter.calls[0]["max_tokens"] == 512


@pytest.mark.asyncio
async def test_response_normalization_and_execution_metadata() -> None:
    fabric, _, _ = await make_fabric()

    response = await fabric.execute_for_role(CognitionRole.JUDGE, "judge candidate patches", owner="judge-owner")

    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 64
    assert response.completion_tokens == 32
    assert response.metadata["runtime_id"] == "judge-runtime"
    assert response.metadata["policy"]["temperature"] == 0.0
    assert response.metadata["future_hooks"]["streaming"] is False


@pytest.mark.asyncio
async def test_runtime_activation_semantics_preserve_bounded_heavyweight_behavior() -> None:
    fabric, lifecycle, _ = await make_fabric()

    await fabric.execute_for_role(CognitionRole.PRIMARY_CODER, "implement feature", owner="coder-owner")
    await fabric.execute_for_role(CognitionRole.JUDGE, "judge candidate patches", owner="judge-owner")

    assert await lifecycle.runtime_health("coder-runtime") is RuntimeState.INACTIVE
    assert await lifecycle.runtime_health("judge-runtime") is RuntimeState.ACTIVE


@pytest.mark.asyncio
async def test_deterministic_execution_and_stable_cognition_responses() -> None:
    fabric, _, _ = await make_fabric()
    request = CognitionRequest(
        task="implement deterministic feature",
        requested_role=CognitionRole.PRIMARY_CODER,
        owner="task-9",
    )

    first = await fabric.execute(request)
    second = await fabric.execute(request)

    assert first.content == second.content
    assert first.runtime_id == second.runtime_id
    assert first.policy == second.policy

