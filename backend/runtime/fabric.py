"""Unified cognition execution orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace

from backend.runtime.binding import BindingDecision, RuntimeBinder
from backend.runtime.cognition import CognitionAdapter
from backend.runtime.lifecycle import CognitionLifecycleManager, RuntimeState
from backend.runtime.routing import CognitionRole, CognitionRouter, RoutingDecision


@dataclass(slots=True)
class CognitionRequest:
    task: str | dict[str, object]
    requested_role: CognitionRole | str | None = None
    repository_context: object | None = None
    patch_summaries: tuple[str, ...] = ()
    execution_summaries: tuple[str, ...] = ()
    critiques: tuple[str, ...] = ()
    architecture_notes: tuple[str, ...] = ()
    memory_summaries: tuple[str, ...] = ()
    requested_tokens: int | None = None
    agent_id: str | None = None
    owner: str | None = None
    system_hint: str | None = None
    policy_override: "InferencePolicy | None" = None


@dataclass(slots=True)
class InferencePolicy:
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_depth: str | None = None
    critique_intensity: str | None = None
    execution_focus: float | None = None
    architecture_focus: float | None = None


@dataclass(slots=True)
class CognitionResponse:
    content: str
    role: CognitionRole
    runtime_id: str
    model: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    policy: InferencePolicy
    metadata: dict[str, object]


class CognitionExecutionFabric:
    """Coordinates role-aware cognition execution over bound runtimes."""

    _DEFAULT_POLICIES: dict[CognitionRole, InferencePolicy] = {
        CognitionRole.PRIMARY_CODER: InferencePolicy(
            temperature=0.2,
            max_tokens=1400,
            reasoning_depth="medium",
            critique_intensity="low",
            execution_focus=0.5,
            architecture_focus=0.4,
        ),
        CognitionRole.JUDGE: InferencePolicy(
            temperature=0.0,
            max_tokens=1200,
            reasoning_depth="high",
            critique_intensity="high",
            execution_focus=0.8,
            architecture_focus=0.7,
        ),
        CognitionRole.RETRY_SPECIALIST: InferencePolicy(
            temperature=0.1,
            max_tokens=1100,
            reasoning_depth="high",
            critique_intensity="medium",
            execution_focus=0.9,
            architecture_focus=0.5,
        ),
        CognitionRole.ARCHITECT: InferencePolicy(
            temperature=0.15,
            max_tokens=1500,
            reasoning_depth="high",
            critique_intensity="medium",
            execution_focus=0.3,
            architecture_focus=0.95,
        ),
        CognitionRole.SYNTHESIZER: InferencePolicy(
            temperature=0.1,
            max_tokens=900,
            reasoning_depth="medium",
            critique_intensity="low",
            execution_focus=0.2,
            architecture_focus=0.6,
        ),
    }

    def __init__(
        self,
        *,
        binder: RuntimeBinder,
        router: CognitionRouter,
        lifecycle_manager: CognitionLifecycleManager,
        cognition_adapter: CognitionAdapter,
    ) -> None:
        self._binder = binder
        self._router = router
        self._lifecycle_manager = lifecycle_manager
        self._cognition_adapter = cognition_adapter

    async def execute(self, request: CognitionRequest) -> CognitionResponse:
        binding_decision, routing_decision, policy = await self.prepare_execution(request)
        activated_decision = await self.activate_runtime(binding_decision, owner=request.owner)
        raw_response = await self._cognition_adapter.complete(
            system_prompt=self._system_prompt(
                activated_decision.role,
                policy,
                system_hint=request.system_hint,
            ),
            user_prompt=routing_decision.prompt_context,
            model_role=activated_decision.runtime_role,
            temperature=policy.temperature or 0.0,
            max_tokens=policy.max_tokens or 0,
            agent_id=request.agent_id,
        )
        return self.normalize_response(
            raw_response,
            binding_decision=activated_decision,
            routing_decision=routing_decision,
            policy=policy,
        )

    async def execute_for_role(
        self,
        role: CognitionRole | str,
        task: str | dict[str, object],
        **kwargs: object,
    ) -> CognitionResponse:
        return await self.execute(
            CognitionRequest(
                task=task,
                requested_role=role,
                **kwargs,
            )
        )

    async def prepare_execution(
        self,
        request: CognitionRequest,
    ) -> tuple[BindingDecision, RoutingDecision, InferencePolicy]:
        binding_decision = await self._binder.dispatch_runtime(
            request.task,
            requested_role=request.requested_role,
            owner=request.owner,
            activate=False,
            repository_context=request.repository_context,
            patch_summaries=request.patch_summaries,
            execution_summaries=request.execution_summaries,
            critiques=request.critiques,
            architecture_notes=request.architecture_notes,
            memory_summaries=request.memory_summaries,
            requested_tokens=request.requested_tokens,
        )
        routing_decision = binding_decision.routing_decision or await self._router.route_task(
            request.task,
            requested_role=request.requested_role,
            repository_context=request.repository_context,
            patch_summaries=request.patch_summaries,
            execution_summaries=request.execution_summaries,
            critiques=request.critiques,
            architecture_notes=request.architecture_notes,
            memory_summaries=request.memory_summaries,
            requested_tokens=request.requested_tokens,
            owner=request.owner,
        )
        policy = self.apply_policy(
            routing_decision.role,
            override=request.policy_override,
        )
        return binding_decision, routing_decision, policy

    async def activate_runtime(
        self,
        binding_decision: BindingDecision,
        *,
        owner: str | None = None,
    ) -> BindingDecision:
        if binding_decision.runtime_id is None:
            raise RuntimeError(f"No runtime available for role '{binding_decision.role.value}'.")
        runtime = await self._lifecycle_manager.activate_runtime(
            binding_decision.runtime_id,
            owner=owner or binding_decision.owner,
        )
        return replace(
            binding_decision,
            owner=runtime.owner,
            health=runtime.health,
        )

    def apply_policy(
        self,
        role: CognitionRole,
        *,
        override: InferencePolicy | None = None,
    ) -> InferencePolicy:
        default_policy = self._DEFAULT_POLICIES[role]
        if override is None:
            return default_policy
        return InferencePolicy(
            temperature=override.temperature if override.temperature is not None else default_policy.temperature,
            max_tokens=override.max_tokens if override.max_tokens is not None else default_policy.max_tokens,
            reasoning_depth=override.reasoning_depth or default_policy.reasoning_depth,
            critique_intensity=override.critique_intensity or default_policy.critique_intensity,
            execution_focus=override.execution_focus if override.execution_focus is not None else default_policy.execution_focus,
            architecture_focus=override.architecture_focus if override.architecture_focus is not None else default_policy.architecture_focus,
        )

    def normalize_response(
        self,
        raw_response: object,
        *,
        binding_decision: BindingDecision,
        routing_decision: RoutingDecision,
        policy: InferencePolicy,
    ) -> CognitionResponse:
        content = getattr(raw_response, "content")
        model = getattr(raw_response, "model")
        prompt_tokens = int(getattr(raw_response, "prompt_tokens"))
        completion_tokens = int(getattr(raw_response, "completion_tokens"))
        finish_reason = getattr(raw_response, "finish_reason")
        metadata = self.execution_metadata(
            binding_decision=binding_decision,
            routing_decision=routing_decision,
            policy=policy,
            raw_response=raw_response,
        )
        return CognitionResponse(
            content=content,
            role=binding_decision.role,
            runtime_id=binding_decision.runtime_id or "",
            model=model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            policy=policy,
            metadata=metadata,
        )

    def execution_metadata(
        self,
        *,
        binding_decision: BindingDecision,
        routing_decision: RoutingDecision,
        policy: InferencePolicy,
        raw_response: object,
    ) -> dict[str, object]:
        return {
            "runtime_id": binding_decision.runtime_id,
            "runtime_role": binding_decision.runtime_role.value if binding_decision.runtime_role is not None else None,
            "runtime_health": binding_decision.health.value if binding_decision.health is not None else None,
            "role": binding_decision.role.value,
            "owner": binding_decision.owner,
            "used_fallback": binding_decision.used_fallback,
            "degraded": binding_decision.degraded,
            "model": getattr(raw_response, "model"),
            "policy": {
                "temperature": policy.temperature,
                "max_tokens": policy.max_tokens,
                "reasoning_depth": policy.reasoning_depth,
                "critique_intensity": policy.critique_intensity,
                "execution_focus": policy.execution_focus,
                "architecture_focus": policy.architecture_focus,
            },
            "routing": {
                "budget_tokens": routing_decision.budget.allocated_tokens,
                "context_truncated": routing_decision.context.truncated,
                "compression_hooks": list(routing_decision.budget.compression_hooks),
            },
            "future_hooks": {
                "streaming": False,
                "tool_use": False,
                "structured_output": False,
                "memory_injection": False,
                "multi_runtime_concurrency": False,
            },
        }

    @staticmethod
    def _system_prompt(
        role: CognitionRole,
        policy: InferencePolicy,
        *,
        system_hint: str | None = None,
    ) -> str:
        base = (
            f"You are Forge acting as {role.value}. "
            f"Reasoning depth={policy.reasoning_depth}; "
            f"critique intensity={policy.critique_intensity}; "
            f"execution focus={policy.execution_focus}; "
            f"architecture focus={policy.architecture_focus}. "
            "Prefer deterministic, architecture-consistent outputs."
        )
        if system_hint:
            return f"{base}\nAdditional directive: {system_hint}"
        return base
