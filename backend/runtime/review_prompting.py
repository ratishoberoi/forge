"""Prompt builders for patch judging and retry guidance."""

from __future__ import annotations

from textwrap import dedent

from backend.runtime.candidate import PatchCandidate


def build_patch_judge_prompt(*, task: str, candidate: PatchCandidate, repository_context: str) -> str:
    return dedent(
        f"""
        You are Forge's patch judge.

        TASK:
        {task}

        REPOSITORY CONTEXT:
        {repository_context}

        CANDIDATE PATCH ID:
        {candidate.candidate_id}

        PATCH SUMMARY:
        {candidate.patch.summary or candidate.patch.title}

        PATCH REASONING:
        {candidate.patch.reasoning or "No explicit reasoning provided."}

        UNIFIED DIFF:
        {candidate.patch.unified_diff}

        Review critically. Prefer minimal, safe diffs that preserve architecture consistency.
        Explicitly analyze:
        - correctness
        - architecture consistency
        - safety
        - minimality
        - maintainability
        - hallucination risk

        Do not assume missing files or APIs exist. Call out uncertainty explicitly.
        """
    ).strip()


def build_candidate_comparison_prompt(*, task: str, candidates: list[PatchCandidate], repository_context: str) -> str:
    candidate_sections = []
    for candidate in candidates:
        candidate_sections.append(
            dedent(
                f"""
                CANDIDATE {candidate.candidate_id}
                Summary: {candidate.patch.summary or candidate.patch.title}
                Reasoning: {candidate.patch.reasoning or "No explicit reasoning provided."}
                Impacted files: {[target.path for target in candidate.patch.impacted_files]}
                Risk: {candidate.patch.risk.value}
                Diff:
                {candidate.patch.unified_diff}
                """
            ).strip()
        )

    return dedent(
        f"""
        Compare the following patch candidates for the task below.

        TASK:
        {task}

        REPOSITORY CONTEXT:
        {repository_context}

        REQUIREMENTS:
        - prefer the smallest safe patch
        - reject architecture drift
        - penalize hallucinated symbols or files
        - compare impacted files explicitly
        - explain tradeoffs between candidates

        {'\n\n'.join(candidate_sections)}
        """
    ).strip()


def build_retry_prompt(*, task: str, critique: str, repository_context: str) -> str:
    return dedent(
        f"""
        A previous patch attempt requires revision.

        TASK:
        {task}

        REPOSITORY CONTEXT:
        {repository_context}

        CRITIQUE:
        {critique}

        Produce a safer, more minimal retry.
        Preserve architecture consistency, avoid unrelated edits, and do not invent nonexistent APIs.
        If uncertainty remains, reduce change scope instead of guessing.
        """
    ).strip()
