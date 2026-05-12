from __future__ import annotations

import asyncio

from backend.runtime.context import SharedContextStore
from backend.runtime.messages import AgentMessage, ContextPayload, MessageKind
from backend.runtime.results import CandidateSolution, ResultAggregator


def test_shared_context_store_snapshots() -> None:
    async def scenario() -> None:
        store = SharedContextStore()
        await store.put_snapshot("task-1", "note", "remember this")
        snapshots = await store.get_snapshots("task-1")
        assert len(snapshots) == 1
        context = await store.build_context("task-1", "inspect auth", summary="inspect auth")
        assert context.estimated_tokens > 0

    asyncio.run(scenario())


def test_result_aggregator_picks_highest_score() -> None:
    aggregator = ResultAggregator()
    low = CandidateSolution(
        task_id="task-1",
        agent_id="agent-low",
        score=0.1,
        confidence=0.1,
        message=AgentMessage(
            id="m1",
            kind=MessageKind.CONTEXT,
            sender="agent-low",
            created_at=0.0,
            payload=ContextPayload(summary="low"),
        ),
    )
    high = CandidateSolution(
        task_id="task-1",
        agent_id="agent-high",
        score=0.9,
        confidence=0.9,
        message=AgentMessage(
            id="m2",
            kind=MessageKind.CONTEXT,
            sender="agent-high",
            created_at=0.0,
            payload=ContextPayload(summary="high"),
        ),
    )
    aggregator.add(low)
    aggregator.add(high)
    aggregated = aggregator.aggregate("task-1")
    assert aggregated.winner is not None
    assert aggregated.winner.agent_id == "agent-high"
