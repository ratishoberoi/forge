"""Developer-facing runtime facade."""

from __future__ import annotations

from backend.config.settings import Settings, get_settings
from backend.repointel.service import RepositoryIntelligenceEngine
from backend.runtime.context import SharedContextStore
from backend.runtime.events import AsyncEventBus
from backend.runtime.mock_agents import (
    MockCoderAgent,
    MockContextAgent,
    MockCriticAgent,
    MockPlannerAgent,
)
from backend.runtime.orchestrator import Orchestrator
from backend.runtime.results import ResultAggregator


class MultiAgentRuntime:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.repo_intelligence = RepositoryIntelligenceEngine(self.settings)
        self.shared_context = SharedContextStore(self.settings, self.repo_intelligence)
        self.event_bus = AsyncEventBus(max_queue_size=self.settings.runtime_event_queue_size)
        self.result_aggregator = ResultAggregator()
        self.orchestrator = Orchestrator(
            self.settings,
            shared_context=self.shared_context,
            event_bus=self.event_bus,
            result_aggregator=self.result_aggregator,
        )

    def register_default_mock_agents(self) -> None:
        self.orchestrator.register_agent(MockPlannerAgent(self.repo_intelligence))
        self.orchestrator.register_agent(MockCoderAgent())
        self.orchestrator.register_agent(MockCriticAgent())
        self.orchestrator.register_agent(MockContextAgent())

    async def start(self) -> None:
        await self.orchestrator.start()

    async def stop(self) -> None:
        await self.orchestrator.stop()
        await self.repo_intelligence.shutdown()
