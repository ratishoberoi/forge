import pytest
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_context import ArtifactContextBuilder
from backend.runtime.artifact_loader import ArtifactLoader
from backend.runtime.artifact_store import ArtifactStore
from backend.runtime.autonomous_artifacts import AutonomousArtifactOrchestrator


# ── Fake adapter ─────────────────────────────────────────────────────────────

class FakeAdapter:
    """Records all execute() calls for inspection."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(
        self,
        *,
        role: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append({
            "role": role,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return f"{role} RESPONSE"


# ── Fixture ───────────────────────────────────────────────────────────────────

def make_orchestrator(tmp_path, adapter: FakeAdapter | None = None) -> tuple[AutonomousArtifactOrchestrator, FakeAdapter]:
    adapter = adapter or FakeAdapter()
    orchestrator = AutonomousArtifactOrchestrator(
        artifact_store=ArtifactStore(base_dir=str(tmp_path)),
        artifact_loader=ArtifactLoader(base_dir=str(tmp_path)),
        context_builder=ArtifactContextBuilder(),
        adapter=adapter,
    )
    return orchestrator, adapter


# ── execute_round ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_round_creates_artifact(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)

    artifact = await orchestrator.execute_round(
        role="PRIMARY_CODER",
        round_id=1,
        task="Add typing",
        prompt="Implement typing safely.",
    )

    assert artifact.role == "PRIMARY_CODER"
    assert artifact.round_id == 1
    assert "RESPONSE" in artifact.content


@pytest.mark.asyncio
async def test_execute_round_persists_to_store(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)

    await orchestrator.execute_round(
        role="PRIMARY_CODER",
        round_id=1,
        task="Add typing",
        prompt="Implement typing safely.",
    )

    store = ArtifactStore(base_dir=str(tmp_path))
    assert store.exists("PRIMARY_CODER", 1)


@pytest.mark.asyncio
async def test_execute_round_artifact_has_metadata(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)

    artifact = await orchestrator.execute_round(
        role="PRIMARY_CODER",
        round_id=1,
        task="Add typing",
        prompt="Implement typing safely.",
        temperature=0.5,
        max_tokens=512,
    )

    assert artifact.metadata["prompt"] == "Implement typing safely."
    assert artifact.metadata["temperature"] == 0.5
    assert artifact.metadata["max_tokens"] == 512


@pytest.mark.asyncio
async def test_execute_round_passes_system_prompt(tmp_path):
    orchestrator, adapter = make_orchestrator(tmp_path)

    await orchestrator.execute_round(
        role="PRIMARY_CODER",
        round_id=1,
        task="Add typing",
        prompt="Implement typing safely.",
        system_prompt="You are a senior Python engineer.",
    )

    assert adapter.calls[0]["system_prompt"] == "You are a senior Python engineer."


@pytest.mark.asyncio
async def test_prior_context_included_in_second_round(tmp_path):
    orchestrator, adapter = make_orchestrator(tmp_path)

    await orchestrator.execute_round(
        role="PRIMARY_CODER",
        round_id=1,
        task="Add typing",
        prompt="Initial implementation",
    )

    await orchestrator.execute_round(
        role="PRIMARY_CODER",
        round_id=2,
        task="Add typing",
        prompt="Refine implementation",
    )

    second_prompt = adapter.calls[1]["prompt"]
    assert "PRIOR COGNITION" in second_prompt
    assert "PRIMARY_CODER RESPONSE" in second_prompt


@pytest.mark.asyncio
async def test_no_prior_context_on_first_round(tmp_path):
    orchestrator, adapter = make_orchestrator(tmp_path)

    await orchestrator.execute_round(
        role="PRIMARY_CODER",
        round_id=1,
        task="Add typing",
        prompt="Initial implementation",
    )

    first_prompt = adapter.calls[0]["prompt"]
    assert "PRIOR COGNITION" not in first_prompt


# ── execute_multi_role_round ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_multi_role_round(tmp_path):
    orchestrator, adapter = make_orchestrator(tmp_path)

    artifacts = await orchestrator.execute_multi_role_round(
        round_id=1,
        task="Add typing",
        prompt="Implement typing safely.",
        roles=["PRIMARY_CODER", "JUDGE"],
    )

    assert len(artifacts) == 2
    assert artifacts[0].role == "PRIMARY_CODER"
    assert artifacts[1].role == "JUDGE"
    assert len(adapter.calls) == 2


# ── execute_judge_round ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_judge_round(tmp_path):
    orchestrator, adapter = make_orchestrator(tmp_path)

    coder_artifact = CognitionArtifact.create(
        role="PRIMARY_CODER",
        round_id=1,
        task="Add typing",
        content="def hello(name: str) -> str: return 'hello ' + name",
    )

    judge_artifact = await orchestrator.execute_judge_round(
        round_id=2,
        task="Add typing",
        coder_artifact=coder_artifact,
    )

    assert judge_artifact.role == "JUDGE"
    assert judge_artifact.round_id == 2
    judge_prompt = adapter.calls[0]["prompt"]
    assert "CODER OUTPUT" in judge_prompt
    assert "def hello" in judge_prompt


@pytest.mark.asyncio
async def test_execute_judge_round_uses_zero_temperature(tmp_path):
    orchestrator, adapter = make_orchestrator(tmp_path)

    coder_artifact = CognitionArtifact.create(
        role="PRIMARY_CODER",
        round_id=1,
        task="Add typing",
        content="some code",
    )

    await orchestrator.execute_judge_round(
        round_id=2,
        task="Add typing",
        coder_artifact=coder_artifact,
    )

    assert adapter.calls[0]["temperature"] == 0.0