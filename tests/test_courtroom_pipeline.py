import pytest
from backend.runtime.courtroom_executor import CourtroomExecutor
from backend.runtime.courtroom_pipeline import CourtroomPipeline
from backend.runtime.courtroom_roles import CourtroomRole
from backend.runtime.courtroom_message import CourtroomMessage


# ── Fixtures & Helpers ─────────────────────────────────────────────────────

@pytest.fixture
def executor() -> CourtroomExecutor:
    return CourtroomExecutor()


@pytest.fixture
def pipeline() -> CourtroomPipeline:
    return CourtroomPipeline()


# ── Tests ──────────────────────────────────────────────────────────────────

def test_pipeline_creation():
    p = CourtroomPipeline()
    assert p.message_count == 0
    assert p.last_message is None


def test_add_message(executor: CourtroomExecutor, pipeline: CourtroomPipeline):
    msg = executor.coder_message("Initial patch v1")
    executor.append(pipeline=pipeline, message=msg)

    assert pipeline.message_count == 1
    assert pipeline.last_message.role == CourtroomRole.PRIMARY_CODER
    assert pipeline.last_message.content == "Initial patch v1"


def test_convenience_role_methods(executor: CourtroomExecutor, pipeline: CourtroomPipeline):
    executor.add_coder_message(pipeline, "Patch")
    executor.add_synth_message(pipeline, "Critique on blast radius")
    executor.add_judge_message(pipeline, "Accepted")

    assert pipeline.message_count == 3
    assert len(pipeline.get_messages_by_role(CourtroomRole.PRIMARY_CODER)) == 1
    assert len(pipeline.get_messages_by_role(CourtroomRole.DEEPSEEK_SYNTH)) == 1
    assert len(pipeline.get_messages_by_role(CourtroomRole.JUDGE)) == 1


def test_full_round_convenience(executor: CourtroomExecutor):
    pipeline = executor.create_pipeline()
    executor.run_full_round(
        pipeline=pipeline,
        coder_patch="def auth(): ...",
        synth_critique="Potential SQL injection risk",
        coder_revision="def auth(): ... (fixed)",
        judge_verdict="Converged. Safe to merge.",
    )

    assert pipeline.message_count == 4
    roles = [m.role for m in pipeline.messages]
    assert roles == [
        CourtroomRole.PRIMARY_CODER,
        CourtroomRole.DEEPSEEK_SYNTH,
        CourtroomRole.PRIMARY_CODER,
        CourtroomRole.JUDGE,
    ]


def test_message_validation():
    with pytest.raises(ValueError, match="blank"):
        CourtroomMessage(
            role=CourtroomRole.PRIMARY_CODER,
            content="   ",
        )


def test_to_dict():
    pipeline = CourtroomPipeline()
    pipeline.add_coder_message("Test patch")   # This works because we have it on Pipeline too
    
    data = pipeline.to_dict()
    assert "pipeline_id" in data
    assert "messages" in data
    assert len(data["messages"]) == 1