from __future__ import annotations
from backend.runtime.execution_result import ExecutionResult


class ExecutionFeedback:
    """
    Converts execution results into autonomous retry context.
    Responsibilities:
    - produce LLM-injectable feedback strings
    - truncate oversized output
    - classify failure type
    - support structured context building
    """

    DEFAULT_MAX_CHARS = 4_000

    def build(
        self,
        result: ExecutionResult,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """
        Build feedback string from ExecutionResult.
        On success: brief confirmation.
        On failure: stdout + stderr for LLM retry context.
        """
        if result.succeeded:
            return self._build_success(result)
        return self._build_failure(result, max_chars=max_chars)

    def build_retry_prompt(
        self,
        result: ExecutionResult,
        *,
        original_objective: str,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """
        Build a full LLM retry prompt from failed execution.
        Injects objective + failure feedback.
        """
        feedback = self.build(result, max_chars=max_chars)
        return (
            f"OBJECTIVE:\n{original_objective}\n\n"
            f"EXECUTION FEEDBACK:\n{feedback}\n\n"
            "Revise the implementation to fix the execution failure."
        )

    def classify(self, result: ExecutionResult) -> str:
        """
        Classify failure type from stderr/stdout.
        Returns one of: 'syntax_error', 'import_error',
        'runtime_error', 'timeout', 'test_failure', 'unknown'.
        """
        if result.succeeded:
            return "success"

        combined = (result.stderr + result.stdout).lower()

        if "syntaxerror" in combined:
            return "syntax_error"
        if "importerror" in combined or "modulenotfounderror" in combined:
            return "import_error"
        if "timed out" in combined:
            return "timeout"
        if "failed" in combined and "test" in combined:
            return "test_failure"
        if "error" in combined or "exception" in combined:
            return "runtime_error"

        return "unknown"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_success(result: ExecutionResult) -> str:
        parts = ["Execution succeeded."]
        if result.has_stdout:
            parts.append(f"OUTPUT:\n{result.stdout.strip()}")
        parts.append(f"Duration: {result.duration_seconds:.2f}s")
        return "\n\n".join(parts)

    @staticmethod
    def _build_failure(
        result: ExecutionResult,
        *,
        max_chars: int,
    ) -> str:
        stdout = result.stdout[:max_chars] if result.stdout else "(empty)"
        stderr = result.stderr[:max_chars] if result.stderr else "(empty)"

        return (
            f"Execution failed (exit code {result.return_code}).\n\n"
            f"COMMAND:\n{result.command_str}\n\n"
            f"STDOUT:\n{stdout}\n\n"
            f"STDERR:\n{stderr}"
        )