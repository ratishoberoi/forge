from __future__ import annotations

from backend.runtime import (
    BaseAgent,
    ContextPayload,
    MultiAgentRuntime,
    Orchestrator,
    SharedContextStore,
    Task,
)


def test_runtime_public_imports_are_stable() -> None:
    assert BaseAgent is not None
    assert ContextPayload is not None
    assert MultiAgentRuntime is not None
    assert Orchestrator is not None
    assert SharedContextStore is not None
    assert Task is not None


def test_runtime_default_constructors_are_usable() -> None:
    runtime = MultiAgentRuntime()
    orchestrator = Orchestrator()
    shared_context = SharedContextStore()

    assert runtime is not None
    assert orchestrator is not None
    assert shared_context is not None
