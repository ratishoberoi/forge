import os
import subprocess
from pathlib import Path
import pytest
from backend.runtime.coder_agent import AutonomousCoderAgent, CodingTask
import json

class FakeLLMService:
    async def generate_text(self, prompt: str, temperature: float = 0.1) -> str:
        return """
{
  "summary": "Add hello endpoint",
  "reasoning": "Created a simple hello API.",
  "risk": "low",
  "files": {
    "app.py": "def hello():\\n    return 'hello world'\\n"
  }
}
"""


class FakeRepoIntel:
    async def build_context(self, query: str) -> str:
        return "Repository contains simple Python services."


class FlakyLLMService:
    def __init__(self):
        self.calls = 0

    async def generate_text(self, prompt: str, temperature: float = 0.1) -> str:
        self.calls += 1
        if self.calls == 1:
            return '{"summary": "x", "reasoning": "x", "risk": "low", "files": {}}'
        content = f"x = {self.calls}"
        return json.dumps({
            "summary": "x",
            "reasoning": "x", 
            "risk": "low",
            "files": {"app.py": content + "\n"}
        })

@pytest.fixture()
def git_env(tmp_path: Path) -> dict[str, str]:
    return {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@test.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@test.com",
        "HOME": str(tmp_path),
        "PATH": os.environ["PATH"],
    }


@pytest.fixture()
def repo(tmp_path: Path, git_env: dict[str, str]) -> Path:
    repo_path = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo_path)], check=True, env=git_env)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=repo_path,
        check=True,
        env=git_env,
    )
    return repo_path


@pytest.mark.asyncio
async def test_autonomous_coder_agent(repo: Path):
    agent = AutonomousCoderAgent(
        repo_root=str(repo),
        llm_service=FakeLLMService(),
        repo_intel=FakeRepoIntel(),
    )
    task = CodingTask(task_id="task-1", prompt="Add hello endpoint")
    result = await agent.execute(task)

    assert result.succeeded is True
    assert result.patch.is_valid()
    assert "diff --git" in result.patch.unified_diff
    assert "hello world" in result.patch.unified_diff
    assert len(result.patch.impacted_files) == 1
    assert result.patch.impacted_files[0].path == "app.py"


@pytest.mark.asyncio
async def test_agent_retries_on_invalid_patch(repo: Path):
    llm = FlakyLLMService()
    agent = AutonomousCoderAgent(
        repo_root=str(repo),
        llm_service=llm,
        repo_intel=FakeRepoIntel(),
    )
    result = await agent.execute(CodingTask(task_id="task-2", prompt="test"))

    assert result.succeeded is True
    assert result.attempts == 2