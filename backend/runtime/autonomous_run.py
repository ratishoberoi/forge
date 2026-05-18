from __future__ import annotations
from pathlib import Path
from backend.runtime.artifact_exchange import ArtifactExchange
from backend.runtime.artifact_loader import ArtifactLoader
from backend.runtime.artifact_store import ArtifactStore
from backend.runtime.autonomous_courtroom import AutonomousCourtroom
from backend.runtime.convergence_loop import ConvergenceLoop, ConvergenceResult
from backend.runtime.execution_feedback import ExecutionFeedback
from backend.runtime.execution_result import ExecutionResult
from backend.runtime.execution_runner import ExecutionRunner
from backend.runtime.git_diff import GitDiff
from backend.runtime.local_inference import LocalInference
from backend.runtime.patch_parser import PatchParser
from backend.runtime.patch_writer import PatchWriter, PatchResult
from backend.runtime.repo_workspace import RepositoryWorkspace
from backend.runtime.revision_judge import RevisionJudge
from backend.runtime.revision_prompt import RevisionPromptBuilder
from backend.runtime.runtime_launcher import RuntimeLauncher
from backend.runtime.runtime_shutdown import RuntimeShutdown
from backend.runtime.runtime_swap_engine import RuntimeSwapEngine


class AutonomousRunError(Exception):
    """Raised when AutonomousRun encounters an unrecoverable error."""


class AutonomousRun:
    """
    Integrated autonomous repository execution.
    Full pipeline:
    1. Convergence loop generates cognition artifacts
    2. Patch parser extracts clean code from model response
    3. Patch written to target file
    4. Test command executed against repo
    5. Feedback printed for inspection
    6. Git diff shown for review
    """

    def __init__(
        self,
        *,
        repo_root: str,
        artifact_dir: str,
        inference_timeout: int = 180,
        backup_writes: bool = True,
    ) -> None:
        self.workspace = RepositoryWorkspace(repo_root)

        exchange = ArtifactExchange(
            store=ArtifactStore(artifact_dir),
            loader=ArtifactLoader(artifact_dir),
        )

        swap_engine = RuntimeSwapEngine(
            launcher=RuntimeLauncher(),
            shutdown=RuntimeShutdown(),
        )

        courtroom = AutonomousCourtroom(
            swap_engine=swap_engine,
            exchange=exchange,
            inference=LocalInference(timeout=inference_timeout),
        )

        self.loop = ConvergenceLoop(
            courtroom=courtroom,
            judge=RevisionJudge(),
            prompt_builder=RevisionPromptBuilder(),
        )

        self.writer = PatchWriter(self.workspace, backup=backup_writes)
        self.runner = ExecutionRunner()
        self.feedback = ExecutionFeedback()
        self.git = GitDiff(repo_root=repo_root)
        self.patch_parser = PatchParser()

    def execute(
        self,
        *,
        objective: str,
        target_file: str,
        test_command: list[str],
        max_iterations: int = 3,
    ) -> None:
        """
        Run full autonomous execution pipeline.
        Stages:
        1. Convergence loop — generate + refine patch
        2. Parse clean code from coder artifact
        3. Write patch to target_file
        4. Run test_command
        5. Print feedback + git diff
        """
        self._print_header("AUTONOMOUS RUN START")
        print(f"Objective : {objective}")
        print(f"Target    : {target_file}")
        print(f"Command   : {' '.join(test_command)}")
        print(f"Max iters : {max_iterations}")

        # Stage 1: Convergence loop
        self._print_header("STAGE 1 — COGNITION")
        result = self.loop.run_full(
            objective=objective,
            max_iterations=max_iterations,
        )
        self._print_convergence_summary(result)

        # Stage 2: Parse + write patch
        self._print_header("STAGE 2 — PATCH WRITE")
        patch_result = self._write_patch(result, target_file)
        if not patch_result.success:
            raise AutonomousRunError(
                f"Patch write failed for '{target_file}': {patch_result.error}"
            )
        print(f"Written   : {patch_result.resolved_path}")

        # Stage 3: Execute tests
        self._print_header("STAGE 3 — TEST EXECUTION")
        exec_result = self.runner.run(
            command=test_command,
            cwd=str(self.workspace.repo_root),
        )
        self._print_execution_result(exec_result)

        # Stage 4: Feedback + diff
        self._print_header("EXECUTION FEEDBACK")
        print(self.feedback.build(exec_result))

        self._print_header("GIT DIFF")
        diff = self.git.diff()
        print(diff if diff.strip() else "(no changes detected)")

        self._print_header("RUN COMPLETE")
        status = "PASSED" if exec_result.succeeded else "FAILED"
        convergence = "CONVERGED" if result.converged else "MAX ITERATIONS"
        print(f"Tests     : {status}")
        print(f"Cognition : {convergence}")
        print(f"Rounds    : {result.iterations_run}")
        print(f"Verdict   : {result.final_verdict}")

    def execute_full(
        self,
        *,
        objective: str,
        target_file: str,
        test_command: list[str],
        max_iterations: int = 3,
    ) -> dict:
        """
        Same as execute() but returns structured result dict.
        Useful for programmatic callers and tests.
        """
        convergence = self.loop.run_full(
            objective=objective,
            max_iterations=max_iterations,
        )

        patch_result = self._write_patch(convergence, target_file)

        exec_result = self.runner.run(
            command=test_command,
            cwd=str(self.workspace.repo_root),
        )

        diff = self.git.diff()
        feedback_text = self.feedback.build(exec_result)
        failure_type = self.feedback.classify(exec_result)

        return {
            "converged": convergence.converged,
            "iterations_run": convergence.iterations_run,
            "final_verdict": convergence.final_verdict,
            "patch_success": patch_result.success,
            "patch_path": (
                str(patch_result.resolved_path)
                if patch_result.resolved_path else None
            ),
            "tests_passed": exec_result.succeeded,
            "return_code": exec_result.return_code,
            "feedback": feedback_text,
            "failure_type": failure_type,
            "diff": diff,
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write_patch(
        self,
        result: ConvergenceResult,
        target_file: str,
    ) -> PatchResult:
        """
        Extract coder artifact from latest round,
        parse clean code, write to target file.
        """
        latest_round = result.final_round
        if not latest_round:
            return PatchResult(
                file_path=target_file,
                success=False,
                error="No artifacts in final round.",
            )

        coder_artifact = latest_round[0]
        generated_content = self.patch_parser.extract_code(
            coder_artifact.content
        )

        if not generated_content.strip():
            return PatchResult(
                file_path=target_file,
                success=False,
                error="Patch parser returned empty content.",
            )

        return self.writer.apply(
            file_path=target_file,
            new_content=generated_content,
        )

    @staticmethod
    def _print_header(title: str) -> None:
        bar = "=" * 50
        print(f"\n{bar}")
        print(f"  {title}")
        print(f"{bar}\n")

    @staticmethod
    def _print_convergence_summary(result: ConvergenceResult) -> None:
        print(f"Iterations  : {result.iterations_run}")
        print(f"Converged   : {result.converged}")
        print(f"Verdict     : {result.final_verdict}")
        print(f"Total arts  : {len(result.all_artifacts)}")

    @staticmethod
    def _print_execution_result(result: ExecutionResult) -> None:
        status = "PASSED" if result.succeeded else "FAILED"
        print(f"Status      : {status}")
        print(f"Return code : {result.return_code}")
        print(f"Duration    : {result.duration_seconds:.2f}s")
        if result.has_stdout:
            print(f"STDOUT:\n{result.stdout.strip()}")
        if result.has_stderr:
            print(f"STDERR:\n{result.stderr.strip()}")