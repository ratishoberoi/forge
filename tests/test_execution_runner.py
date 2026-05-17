import pytest
from backend.runtime.execution_feedback import ExecutionFeedback
from backend.runtime.execution_result import ExecutionResult
from backend.runtime.execution_runner import ExecutionRunner, ExecutionRunnerError


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_runner() -> ExecutionRunner:
    return ExecutionRunner()


def make_feedback() -> ExecutionFeedback:
    return ExecutionFeedback()


# ── ExecutionResult ───────────────────────────────────────────────────────────

def test_execution_result_succeeded():
    result = ExecutionResult(
        command=["echo", "hi"],
        return_code=0,
        stdout="hi\n",
        stderr="",
    )
    assert result.succeeded is True
    assert result.failed is False


def test_execution_result_failed():
    result = ExecutionResult(
        command=["false"],
        return_code=1,
        stdout="",
        stderr="error",
    )
    assert result.succeeded is False
    assert result.failed is True


def test_execution_result_command_str():
    result = ExecutionResult(
        command=["python", "-c", "print(1)"],
        return_code=0,
        stdout="1\n",
        stderr="",
    )
    assert result.command_str == "python -c print(1)"


def test_execution_result_has_stdout():
    result = ExecutionResult(
        command=["x"], return_code=0, stdout="output\n", stderr=""
    )
    assert result.has_stdout is True


def test_execution_result_has_stderr():
    result = ExecutionResult(
        command=["x"], return_code=1, stdout="", stderr="error\n"
    )
    assert result.has_stderr is True


def test_execution_result_to_dict():
    result = ExecutionResult(
        command=["python", "-c", "pass"],
        return_code=0,
        stdout="",
        stderr="",
    )
    d = result.to_dict()
    assert d["return_code"] == 0
    assert d["succeeded"] is True
    assert "executed_at" in d


# ── ExecutionRunner ───────────────────────────────────────────────────────────

def test_execution_success():
    runner = make_runner()
    result = runner.run(command=["python", "-c", "print('ok')"])
    assert result.succeeded
    assert "ok" in result.stdout


def test_execution_failure():
    runner = make_runner()
    result = runner.run(command=["python", "-c", "raise ValueError('bad')"])
    assert not result.succeeded
    assert "ValueError" in result.stderr


def test_execution_return_code_zero_on_success():
    runner = make_runner()
    result = runner.run(command=["python", "-c", "pass"])
    assert result.return_code == 0


def test_execution_return_code_nonzero_on_failure():
    runner = make_runner()
    result = runner.run(command=["python", "-c", "import sys; sys.exit(2)"])
    assert result.return_code == 2


def test_execution_duration_recorded():
    runner = make_runner()
    result = runner.run(command=["python", "-c", "pass"])
    assert result.duration_seconds >= 0


def test_execution_empty_command_raises():
    runner = make_runner()
    with pytest.raises(ExecutionRunnerError, match="empty"):
        runner.run(command=[])


def test_execution_unknown_command_raises():
    runner = make_runner()
    with pytest.raises(ExecutionRunnerError, match="not found"):
        runner.run(command=["nonexistent_command_xyz"])


def test_execution_timeout_returns_failed_result():
    runner = ExecutionRunner(timeout=1)
    result = runner.run(
        command=["python", "-c", "import time; time.sleep(10)"],
        timeout=1,
    )
    assert result.failed
    assert "timed out" in result.stderr.lower()


def test_execution_run_script():
    runner = make_runner()
    result = runner.run_script("print('from script')")
    assert result.succeeded
    assert "from script" in result.stdout


def test_execution_with_cwd(tmp_path):
    runner = make_runner()
    result = runner.run(
        command=["python", "-c", "import os; print(os.getcwd())"],
        cwd=str(tmp_path),
    )
    assert result.succeeded
    assert str(tmp_path) in result.stdout


# ── ExecutionFeedback ─────────────────────────────────────────────────────────

def test_execution_feedback_success():
    runner = make_runner()
    feedback = make_feedback()
    result = runner.run(command=["python", "-c", "print('ok')"])
    text = feedback.build(result)
    assert "succeeded" in text.lower()


def test_execution_feedback_failure():
    runner = make_runner()
    feedback = make_feedback()
    result = runner.run(
        command=["python", "-c", "raise RuntimeError('boom')"]
    )
    text = feedback.build(result)
    assert "failed" in text.lower()
    assert "RuntimeError" in text


def test_execution_feedback_success_includes_output():
    runner = make_runner()
    feedback = make_feedback()
    result = runner.run(command=["python", "-c", "print('hello world')"])
    text = feedback.build(result)
    assert "hello world" in text


def test_execution_feedback_failure_includes_command():
    runner = make_runner()
    feedback = make_feedback()
    result = runner.run(
        command=["python", "-c", "raise ValueError('test')"]
    )
    text = feedback.build(result)
    assert "python" in text


def test_execution_feedback_truncation():
    feedback = make_feedback()
    result = ExecutionResult(
        command=["x"],
        return_code=1,
        stdout="A" * 10_000,
        stderr="B" * 10_000,
    )
    text = feedback.build(result, max_chars=100)
    assert len(text) < 10_000


def test_execution_feedback_build_retry_prompt():
    feedback = make_feedback()
    result = ExecutionResult(
        command=["python", "-c", "raise ValueError"],
        return_code=1,
        stdout="",
        stderr="ValueError",
    )
    prompt = feedback.build_retry_prompt(
        result,
        original_objective="Fix the auth module",
    )
    assert "OBJECTIVE:" in prompt
    assert "Fix the auth module" in prompt
    assert "EXECUTION FEEDBACK:" in prompt
    assert "Revise" in prompt


def test_execution_feedback_classify_success():
    feedback = make_feedback()
    result = ExecutionResult(
        command=["x"], return_code=0, stdout="", stderr=""
    )
    assert feedback.classify(result) == "success"


def test_execution_feedback_classify_syntax_error():
    runner = make_runner()
    feedback = make_feedback()
    result = runner.run(command=["python", "-c", "def f(: pass"])
    assert feedback.classify(result) == "syntax_error"


def test_execution_feedback_classify_runtime_error():
    runner = make_runner()
    feedback = make_feedback()
    result = runner.run(
        command=["python", "-c", "raise RuntimeError('boom')"]
    )
    assert feedback.classify(result) == "runtime_error"


def test_execution_feedback_classify_timeout():
    feedback = make_feedback()
    result = ExecutionResult(
        command=["x"],
        return_code=-1,
        stdout="",
        stderr="Command timed out after 1s.",
    )
    assert feedback.classify(result) == "timeout"


def test_execution_feedback_classify_import_error():
    runner = make_runner()
    feedback = make_feedback()
    result = runner.run(
        command=["python", "-c", "import nonexistent_module_xyz"]
    )
    assert feedback.classify(result) == "import_error"