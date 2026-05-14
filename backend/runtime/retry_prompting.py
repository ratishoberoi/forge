"""Prompt builders for retry and self-repair orchestration."""

from __future__ import annotations

from textwrap import dedent

from backend.runtime.candidate import PatchCandidate
from backend.runtime.judge import JudgeResult


def build_retry_candidate_prompt(
    *,
    task: str,
    candidate: PatchCandidate,
    judge_result: JudgeResult,
    repository_context: str,
) -> str:
    return dedent(
        f"""
        A prior patch candidate requires a bounded retry.

        TASK:
        {task}

        REPOSITORY CONTEXT:
        {repository_context}

        CURRENT CANDIDATE:
        {candidate.candidate_id}

        CURRENT PATCH SUMMARY:
        {candidate.patch.summary or candidate.patch.title}

        JUDGE CRITIQUE:
        {judge_result.critique_summary}

        JUDGE REASONING:
        {judge_result.reasoning}

        REQUIREMENTS:
        - keep the retry minimal and safe
        - preserve architecture consistency
        - avoid unnecessary rewrites
        - preserve already-correct code
        - do not hallucinate files, symbols, or fixes
        - improve only the issues identified by the judge
        """
    ).strip()


def build_self_repair_prompt(
    *,
    task: str,
    critique: str,
    repository_context: str,
) -> str:
    return dedent(
        f"""
        Self-repair the patch candidate for the task below.

        TASK:
        {task}

        REPOSITORY CONTEXT:
        {repository_context}

        CRITIQUE:
        {critique}

        Produce the smallest safe repair that resolves the critique.
        Preserve architecture, keep working code intact, and avoid speculative or hallucinated fixes.
        """
    ).strip()


def build_convergence_warning_prompt(
    *,
    task: str,
    retry_count: int,
    convergence_reason: str,
) -> str:
    return dedent(
        f"""
        Retry convergence warning.

        TASK:
        {task}

        RETRY COUNT:
        {retry_count}

        CONVERGENCE STATUS:
        {convergence_reason}

        Stop broad rewriting. If another attempt is made, it must be narrowly targeted,
        architecture-aware, and strictly safer than previous retries.
        """
    ).strip()
