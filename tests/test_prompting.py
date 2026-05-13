from backend.runtime.prompting import (
    build_coder_system_prompt,
    build_patch_generation_prompt,
    build_patch_review_prompt,
)


def test_system_prompt():
    prompt = build_coder_system_prompt()
    assert "Forge" in prompt
    assert "autonomous" in prompt
    assert "hallucinating" in prompt
    assert "production-grade" in prompt


def test_patch_generation_prompt():
    prompt = build_patch_generation_prompt(
        task="Add JWT auth",
        repository_context="auth.py",
    )
    assert "Add JWT auth" in prompt
    assert "auth.py" in prompt
    assert "JSON" in prompt
    assert "files" in prompt


def test_patch_review_prompt():
    prompt = build_patch_review_prompt(
        task="Review patch",
        patch="diff --git",
    )
    assert "diff --git" in prompt
    assert "security risks" in prompt
    assert "approval recommendation" in prompt


def test_prompts_are_stripped():
    """No leading or trailing whitespace on any prompt."""
    assert build_coder_system_prompt() == build_coder_system_prompt().strip()
    assert build_patch_generation_prompt(task="x", repository_context="y") == \
        build_patch_generation_prompt(task="x", repository_context="y").strip()
    assert build_patch_review_prompt(task="x", patch="y") == \
        build_patch_review_prompt(task="x", patch="y").strip()


def test_patch_generation_interpolation():
    """Task and context must appear verbatim in output."""
    task = "unique-task-abc123"
    ctx = "unique-context-xyz789"
    prompt = build_patch_generation_prompt(task=task, repository_context=ctx)
    assert task in prompt
    assert ctx in prompt