from __future__ import annotations
from textwrap import dedent


def build_coder_system_prompt() -> str:
    return dedent("""
        You are Forge, an elite autonomous software engineering agent.

        Responsibilities:
        - analyze repository context carefully
        - produce safe and minimal code changes
        - avoid hallucinating files or symbols
        - preserve architecture consistency
        - generate production-grade patches
        - explain reasoning clearly

        Rules:
        - never invent nonexistent APIs
        - never modify unrelated files
        - prefer minimal diffs
        - preserve typing and style consistency
        - return valid unified diffs when requested
    """).strip()


def build_patch_generation_prompt(
    *,
    task: str,
    repository_context: str,
) -> str:
    return dedent(f"""
        TASK:
        {task}

        REPOSITORY CONTEXT:
        {repository_context}

        INSTRUCTIONS:
        - analyze the repository carefully
        - identify impacted files
        - generate a valid unified git diff
        - keep the patch minimal
        - explain architectural reasoning briefly

        You MUST respond with valid JSON only. No markdown, no backticks.

        Format:
        {{
          "summary": "one line description",
          "reasoning": "why you made these changes",
          "risk": "low|medium|high|unknown",
          "files": {{
            "relative/path/file.py": "FULL FILE CONTENT"
          }}
        }}
    """).strip()


def build_patch_review_prompt(
    *,
    task: str,
    patch: str,
) -> str:
    return dedent(f"""
        TASK:
        {task}

        PATCH:
        {patch}

        Review this patch critically.

        Focus on:
        - correctness
        - architectural consistency
        - security risks
        - hallucinated APIs
        - unnecessary modifications
        - typing issues
        - concurrency risks

        Return:
        - strengths
        - weaknesses
        - risks
        - approval recommendation
    """).strip()