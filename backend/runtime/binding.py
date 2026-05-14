"""Runtime role binding and cognition dispatch orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.llm.router import ModelRole
from backend.runtime.lifecycle import (
    CognitionLifecycleManager,
    RuntimeEndpoint,
    RuntimeState,
)
from backend.runtime.routing import CognitionRole, CognitionRouter, RoutingDecision


@dataclass(slots=True)
class RuntimeBinding:
    role: CognitionRole
    runtime_id: str
    fallback_runtime_ids: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class BindingDecision:
    role: CognitionRole
    runtime_id: str | None
    endpoint: RuntimeEndpoint | None
    owner: str | None
    health: RuntimeState | None
    used_fallback: bool
    degraded: bool
    runtime_role: ModelRole | None
    binding: RuntimeBinding | None
    routing_decision: RoutingDecision | None = None


class RuntimeBinder:
    """Binds cognition roles to concrete runtimes without changing router semantics."""

    def __init__(
        self,
        *,
        lifecycle_manager: CognitionLifecycleManager,
        router: CognitionRouter | None = None,
    ) -> None:
        self._lifecycle_manager = lifecycle_manager
        self._router = router or CognitionRouter(lifecycle_manager=lifecycle_manager)
        self._bindings: dict[CognitionRole, RuntimeBinding] = {}

    def bind_role(
        self,
        role: CognitionRole | str,
        runtime_id: str,
        *,
        fallback_runtime_ids: tuple[str, ...] | list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeBinding:
        normalized_role = self._normalize_role(role)
        binding = RuntimeBinding(
            role=normalized_role,
            runtime_id=runtime_id,
            fallback_runtime_ids=tuple(fallback_runtime_ids or ()),
            metadata=dict(metadata or {}),
        )
        self._bindings[normalized_role] = binding
        return binding

    def binding_for_role(self, role: CognitionRole | str) -> RuntimeBinding | None:
        return self._bindings.get(self._normalize_role(role))

    async def activate_binding(
        self,
        role: CognitionRole | str,
        *,
        owner: str | None = None,
    ) -> BindingDecision:
        binding = self._require_binding(role)
        target_runtime_id, used_fallback, degraded = await self._resolve_runtime(binding)
        if target_runtime_id is None:
            return BindingDecision(
                role=binding.role,
                runtime_id=None,
                endpoint=None,
                owner=owner,
                health=None,
                used_fallback=used_fallback,
                degraded=degraded,
                runtime_role=self._model_role_for(binding.role),
                binding=binding,
            )
        await self._lifecycle_manager.activate_runtime(target_runtime_id, owner=owner)
        return await self._build_decision(
            binding.role,
            runtime_id=target_runtime_id,
            owner=owner,
            used_fallback=used_fallback,
            degraded=degraded,
            binding=binding,
        )

    async def swap_binding(
        self,
        role: CognitionRole | str,
        next_runtime_id: str,
        *,
        owner: str | None = None,
        fallback_runtime_ids: tuple[str, ...] | list[str] | None = None,
    ) -> BindingDecision:
        normalized_role = self._normalize_role(role)
        current_binding = self._require_binding(normalized_role)
        current_runtime_id, _, _ = await self._resolve_runtime(current_binding)
        updated_binding = RuntimeBinding(
            role=normalized_role,
            runtime_id=next_runtime_id,
            fallback_runtime_ids=tuple(fallback_runtime_ids or current_binding.fallback_runtime_ids),
            metadata=dict(current_binding.metadata),
        )
        self._bindings[normalized_role] = updated_binding

        if current_runtime_id is None:
            await self._lifecycle_manager.activate_runtime(next_runtime_id, owner=owner)
        elif current_runtime_id == next_runtime_id:
            await self._lifecycle_manager.activate_runtime(next_runtime_id, owner=owner)
        else:
            await self._lifecycle_manager.swap_runtime(current_runtime_id, next_runtime_id, owner=owner)

        return await self._build_decision(
            normalized_role,
            runtime_id=next_runtime_id,
            owner=owner,
            used_fallback=False,
            degraded=False,
            binding=updated_binding,
        )

    async def active_binding(self) -> BindingDecision | None:
        active_runtime = await self._lifecycle_manager.active_runtime()
        if active_runtime is None:
            return None
        for binding in self._bindings.values():
            runtime_ids = (binding.runtime_id, *binding.fallback_runtime_ids)
            if active_runtime.runtime_id in runtime_ids:
                return await self._build_decision(
                    binding.role,
                    runtime_id=active_runtime.runtime_id,
                    owner=active_runtime.owner,
                    used_fallback=active_runtime.runtime_id != binding.runtime_id,
                    degraded=active_runtime.runtime_id != binding.runtime_id,
                    binding=binding,
                )
        return BindingDecision(
            role=self._role_for_model_role(active_runtime.role),
            runtime_id=active_runtime.runtime_id,
            endpoint=active_runtime.endpoint,
            owner=active_runtime.owner,
            health=active_runtime.health,
            used_fallback=False,
            degraded=False,
            runtime_role=active_runtime.role,
            binding=None,
        )

    async def dispatch_runtime(
        self,
        task: str | dict[str, object],
        *,
        requested_role: CognitionRole | str | None = None,
        owner: str | None = None,
        activate: bool = False,
        **route_kwargs: object,
    ) -> BindingDecision:
        routing_decision = await self._router.route_task(
            task,
            requested_role=requested_role,
            owner=owner,
            **route_kwargs,
        )
        binding = self.binding_for_role(routing_decision.role)

        if binding is None:
            return BindingDecision(
                role=routing_decision.role,
                runtime_id=routing_decision.runtime_id,
                endpoint=routing_decision.endpoint,
                owner=owner,
                health=await self._health_or_none(routing_decision.runtime_id),
                used_fallback=False,
                degraded=routing_decision.runtime_id is None,
                runtime_role=routing_decision.runtime_role,
                binding=None,
                routing_decision=routing_decision,
            )

        runtime_id, used_fallback, degraded = await self._resolve_runtime(binding)
        if activate and runtime_id is not None:
            await self._lifecycle_manager.activate_runtime(runtime_id, owner=owner)

        return await self._build_decision(
            routing_decision.role,
            runtime_id=runtime_id,
            owner=owner,
            used_fallback=used_fallback,
            degraded=degraded,
            binding=binding,
            routing_decision=routing_decision,
        )

    async def fallback_runtime(self, role: CognitionRole | str) -> BindingDecision | None:
        binding = self.binding_for_role(role)
        if binding is None:
            return None
        for runtime_id in binding.fallback_runtime_ids:
            health = await self._health_or_none(runtime_id)
            if health in {RuntimeState.ACTIVE, RuntimeState.INACTIVE, RuntimeState.LOADING}:
                return await self._build_decision(
                    binding.role,
                    runtime_id=runtime_id,
                    owner=None,
                    used_fallback=True,
                    degraded=True,
                    binding=binding,
                )
        return None

    async def _resolve_runtime(self, binding: RuntimeBinding) -> tuple[str | None, bool, bool]:
        primary_health = await self._health_or_none(binding.runtime_id)
        if primary_health in {RuntimeState.ACTIVE, RuntimeState.INACTIVE, RuntimeState.LOADING}:
            return binding.runtime_id, False, False

        fallback = await self.fallback_runtime(binding.role)
        if fallback is None:
            return None, False, True
        return fallback.runtime_id, True, True

    async def _build_decision(
        self,
        role: CognitionRole,
        *,
        runtime_id: str | None,
        owner: str | None,
        used_fallback: bool,
        degraded: bool,
        binding: RuntimeBinding | None,
        routing_decision: RoutingDecision | None = None,
    ) -> BindingDecision:
        endpoint = None
        health = None
        if runtime_id is not None:
            endpoint = await self._lifecycle_manager.inference_endpoint(runtime_id=runtime_id)
            health = await self._lifecycle_manager.runtime_health(runtime_id)
        return BindingDecision(
            role=role,
            runtime_id=runtime_id,
            endpoint=endpoint,
            owner=owner,
            health=health,
            used_fallback=used_fallback,
            degraded=degraded,
            runtime_role=self._model_role_for(role),
            binding=binding,
            routing_decision=routing_decision,
        )

    async def _health_or_none(self, runtime_id: str | None) -> RuntimeState | None:
        if runtime_id is None:
            return None
        try:
            return await self._lifecycle_manager.runtime_health(runtime_id)
        except KeyError:
            return None

    @staticmethod
    def _normalize_role(role: CognitionRole | str) -> CognitionRole:
        if isinstance(role, CognitionRole):
            return role
        return CognitionRole(role)

    @staticmethod
    def _model_role_for(role: CognitionRole) -> ModelRole:
        return CognitionRouter._ROLE_TO_MODEL_ROLE[role]

    @staticmethod
    def _role_for_model_role(role: ModelRole) -> CognitionRole:
        reverse_mapping = {value: key for key, value in CognitionRouter._ROLE_TO_MODEL_ROLE.items()}
        return reverse_mapping[role]

    def _require_binding(self, role: CognitionRole | str) -> RuntimeBinding:
        normalized_role = self._normalize_role(role)
        try:
            return self._bindings[normalized_role]
        except KeyError as exc:
            raise KeyError(f"No runtime binding registered for role '{normalized_role.value}'.") from exc
