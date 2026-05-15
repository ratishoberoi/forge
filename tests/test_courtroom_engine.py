import pytest

from backend.runtime.courtroom_dispatcher import CourtroomDispatcher
from backend.runtime.courtroom_engine import CourtroomEngine
from backend.runtime.courtroom_roles import CourtroomRole
from backend.runtime.courtroom_runtime import RuntimeCourtroomResponse
from backend.runtime.courtroom_session import CourtroomSession   # ← Added this


@pytest.fixture
def dispatcher() -> CourtroomDispatcher:
    return CourtroomDispatcher()


@pytest.fixture
def engine(dispatcher: CourtroomDispatcher) -> CourtroomEngine:
    return CourtroomEngine(dispatcher)


def test_courtroom_engine_execution(engine: CourtroomEngine):
    session = engine.execute(objective="Improve authentication flow")

    assert session.response_count == 3
    assert session.objective == "Improve authentication flow"

    roles = [r.role for r in session.responses]
    assert roles == [
        CourtroomRole.PRIMARY_CODER,
        CourtroomRole.DEEPSEEK_SYNTH,
        CourtroomRole.JUDGE,
    ]


def test_session_convenience_methods():
    session = CourtroomSession(objective="Test objective")
    
    session.add_coder_response("Initial patch")
    session.add_synth_response("Security concern found")
    session.add_judge_response("Approved after revision")

    assert session.response_count == 3
    assert session.last_response is not None
    assert session.last_response.role == CourtroomRole.JUDGE


def test_response_validation():
    with pytest.raises(ValueError, match="blank"):
        RuntimeCourtroomResponse(
            role=CourtroomRole.PRIMARY_CODER,
            content="   ",
        )