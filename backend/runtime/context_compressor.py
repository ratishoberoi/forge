from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.runtime.architecture_memory import ArchitectureMemoryRecord
from backend.runtime.context_budget import ContextBudgetManager, ContextChunk


@dataclass(slots=True)
class CompressedRepositoryContext:
    objective: str
    architecture_summary: str
    dependency_summary: str
    test_summary: str
    file_summaries: dict[str, str] = field(default_factory=dict)
    selected_files: list[str] = field(default_factory=list)
    token_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "architecture_summary": self.architecture_summary,
            "dependency_summary": self.dependency_summary,
            "test_summary": self.test_summary,
            "file_summaries": self.file_summaries,
            "selected_files": self.selected_files,
            "token_estimate": self.token_estimate,
        }


class ContextCompressor:
    """Builds bounded high-value repository context for long-horizon prompts."""

    def __init__(self, *, max_tokens: int = 6000, max_file_chars: int = 2200) -> None:
        self.budget = ContextBudgetManager(max_tokens=max_tokens)
        self.max_file_chars = max_file_chars

    def compress(
        self,
        *,
        objective: str,
        architecture_summary: str,
        file_summaries: dict[str, str],
        dependency_graph: dict[str, list[str]],
        related_tests: list[str],
        memory: ArchitectureMemoryRecord | None = None,
    ) -> CompressedRepositoryContext:
        selected = self._rank_files(objective, file_summaries, dependency_graph, memory)
        bounded_files: dict[str, str] = {}
        chunks: list[ContextChunk] = [
            ContextChunk(content=architecture_summary, priority=100),
            ContextChunk(content=self._dependency_summary(dependency_graph, selected), priority=80),
            ContextChunk(content=self._test_summary(related_tests), priority=70),
        ]
        if memory:
            chunks.append(
                ContextChunk(
                    content=(
                        "MEMORY\n"
                        f"previously_modified={memory.previously_modified_files}\n"
                        f"recurring_failures={memory.recurring_failure_patterns}"
                    ),
                    priority=60,
                )
            )
        for path in selected:
            summary = self._summarize_file(path, file_summaries.get(path, ""))
            chunks.append(ContextChunk(content=f"FILE {path}\n{summary}", priority=50))
            bounded_files[path] = summary

        combined = self.budget.build_context(chunks)
        return CompressedRepositoryContext(
            objective=objective,
            architecture_summary=architecture_summary,
            dependency_summary=self._dependency_summary(dependency_graph, selected),
            test_summary=self._test_summary(related_tests),
            file_summaries=bounded_files,
            selected_files=selected,
            token_estimate=self.budget.estimate_tokens(combined),
        )

    def _rank_files(
        self,
        objective: str,
        file_summaries: dict[str, str],
        dependency_graph: dict[str, list[str]],
        memory: ArchitectureMemoryRecord | None,
    ) -> list[str]:
        terms = set(re.findall(r"[A-Za-z0-9]+", objective.lower()))
        scores: list[tuple[int, str]] = []
        modified = set(memory.previously_modified_files if memory else [])
        important = set(memory.important_modules if memory else [])
        for path, content in file_summaries.items():
            path_terms = set(re.findall(r"[A-Za-z0-9]+", path.lower()))
            content_terms = set(re.findall(r"[A-Za-z0-9]+", content[:4000].lower()))
            score = len(terms & path_terms) * 5 + len(terms & content_terms)
            score += len(dependency_graph.get(path, [])) * 2
            if path in important:
                score += 5
            if path in modified:
                score += 3
            if "test" in path.lower():
                score += 1
            scores.append((score, path))
        selected = [path for _, path in sorted(scores, key=lambda item: (-item[0], item[1]))[:18]]
        return selected

    def _summarize_file(self, path: str, content: str) -> str:
        lines = content.splitlines()
        important = [
            line
            for line in lines
            if line.strip().startswith(("class ", "def ", "async def ", "function ", "export ", "import ", "from "))
        ][:80]
        summary = "\n".join(important) if important else content[: self.max_file_chars]
        return summary[: self.max_file_chars]

    @staticmethod
    def _dependency_summary(graph: dict[str, list[str]], selected: list[str]) -> str:
        rows = [f"{path} -> {', '.join(graph.get(path, [])[:8]) or 'no local deps'}" for path in selected[:20]]
        return "DEPENDENCIES\n" + "\n".join(rows)

    @staticmethod
    def _test_summary(related_tests: list[str]) -> str:
        return "TESTS\n" + ("\n".join(related_tests[:30]) if related_tests else "No related tests detected.")
