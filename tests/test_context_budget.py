from backend.runtime.context_budget import ContextBudgetManager, ContextChunk


def test_estimate_tokens():
    manager = ContextBudgetManager()
    assert manager.estimate_tokens("a" * 400) >= 100


def test_context_within_budget():
    manager = ContextBudgetManager(max_tokens=100)
    chunks = [
        ContextChunk(content="a" * 100, priority=1),
        ContextChunk(content="b" * 100, priority=2),
    ]
    assert manager.build_context(chunks)


def test_priority_ordering():
    manager = ContextBudgetManager(max_tokens=1000)
    chunks = [
        ContextChunk(content="LOW", priority=1),
        ContextChunk(content="HIGH", priority=10),
    ]
    assert manager.build_context(chunks).startswith("HIGH")


def test_budget_truncation():
    manager = ContextBudgetManager(max_tokens=10)
    chunks = [
        ContextChunk(content="a" * 100, priority=10),
        ContextChunk(content="b" * 100, priority=9),
    ]
    assert manager.build_context(chunks).count("\n\n") <= 1


def test_empty_context():
    assert ContextBudgetManager().build_context([]) == ""


def test_continue_on_oversized_chunk():
    """Large high-priority chunk must not block smaller lower-priority chunks."""
    manager = ContextBudgetManager(max_tokens=10)
    chunks = [
        ContextChunk(content="a" * 400, priority=10),  # too big
        ContextChunk(content="small", priority=5),      # fits
    ]
    context = manager.build_context(chunks)
    assert "small" in context