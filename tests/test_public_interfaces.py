from __future__ import annotations

from backend.repointel import (
    ContextBuilder,
    EmbeddingService,
    PlanningLayer,
    RepositoryIntelligenceDiagnostics,
    RepositoryIntelligenceEngine,
    TreeSitterParser,
)
from backend.repointel.api import HybridRetrievalEngine


def test_public_imports_are_stable() -> None:
    assert EmbeddingService is not None
    assert TreeSitterParser is not None
    assert HybridRetrievalEngine is not None
    assert ContextBuilder is not None
    assert PlanningLayer is not None
    assert RepositoryIntelligenceDiagnostics is not None


def test_default_constructors_are_usable() -> None:
    embedding_service = EmbeddingService()
    parser = TreeSitterParser()
    retrieval = HybridRetrievalEngine()
    context_builder = ContextBuilder()
    planner = PlanningLayer()
    engine = RepositoryIntelligenceEngine()

    assert embedding_service is not None
    assert parser is not None
    assert retrieval is not None
    assert context_builder is not None
    assert planner is not None
    assert engine is not None
