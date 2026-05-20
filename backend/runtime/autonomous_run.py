from __future__ import annotations
from pathlib import Path
import asyncio
from backend.runtime.artifact_exchange import ArtifactExchange
from backend.runtime.artifact_loader import ArtifactLoader
from backend.runtime.artifact_store import ArtifactStore
from backend.runtime.architecture_memory import ArchitectureMemory
from backend.runtime.autonomous_courtroom import AutonomousCourtroom
from backend.runtime.autonomous_repair import (
    AutonomousRepairConvergenceEngine,
    RepairContext,
)
from backend.runtime.convergence_loop import ConvergenceLoop, ConvergenceResult
from backend.runtime.execution_feedback import ExecutionFeedback
from backend.runtime.execution_result import ExecutionResult
from backend.runtime.execution_runner import ExecutionRunner
from backend.runtime.git_manager import GitManager, GitManagerError
from backend.runtime.git_diff import GitDiff
from backend.runtime.local_inference import LocalInference
from backend.runtime.long_horizon import ObjectiveMemory, ObjectiveMemoryRecord
from backend.runtime.patch_parser import PatchParser
from backend.runtime.patch_writer import PatchWriter, PatchResult
from backend.runtime.release_report import ReleaseReportStore, build_release_report
from backend.runtime.repository_bootstrap import RepositoryBootstrap
from backend.runtime.repo_workspace import RepositoryWorkspace
from backend.runtime.repository_execution_engine import (
    RepositoryExecutionEngine,
    RepositoryExecutionPreparation,
)
from backend.runtime.revision_judge import RevisionJudge
from backend.runtime.revision_prompt import RevisionPromptBuilder
from backend.runtime.runtime_launcher import RuntimeLauncher
from backend.runtime.runtime_shutdown import RuntimeShutdown
from backend.runtime.runtime_swap_engine import RuntimeSwapEngine
from backend.runtime.validation_suite import (
    AcceptanceValidator,
    BuildValidator,
    QualityScorer,
    VisualValidator,
    split_commands,
)


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
        self.repository_engine = RepositoryExecutionEngine(repo_root=repo_root)
        self.architecture_memory = ArchitectureMemory()
        self.objective_memory = ObjectiveMemory()
        self.release_reports = ReleaseReportStore()

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
        preparation = self._prepare_repository_execution(objective)
        objective_with_context = self._objective_with_repository_context(
            objective,
            preparation,
        )
        result = self.loop.run_full(
            objective=objective_with_context,
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
        run_id: str | None = None,
    ) -> dict:
        """
        Same as execute() but returns structured result dict.
        Useful for programmatic callers and tests.
        """
        run_identifier = run_id or "manual"
        execution_branch = self._prepare_execution_branch(run_identifier)
        bootstrap_result = RepositoryBootstrap(str(self.workspace.repo_root)).bootstrap_if_needed(objective)
        preparation = self._prepare_repository_execution(objective)
        convergence = self.loop.run_full(
            objective=self._objective_with_repository_context(objective, preparation),
            max_iterations=max_iterations,
        )
        self._ensure_target_allowed(preparation, target_file)

        coder_artifact = convergence.coder_artifacts[-1] if convergence.coder_artifacts else None
        synth_artifact = convergence.synth_artifacts[-1] if convergence.synth_artifacts else None
        if coder_artifact is None:
            raise AutonomousRunError("No PRIMARY_CODER artifact available for patch application.")

        repair_engine = AutonomousRepairConvergenceEngine(
            repo_root=str(self.workspace.repo_root),
            repository_engine=self.repository_engine,
            runner=self.runner,
            repair_generator=self._generate_repair_patch,
        )
        repair_result = repair_engine.run(
            objective=objective,
            preparation=preparation,
            initial_response_text=coder_artifact.content,
            test_command=test_command,
            validation_commands=[test_command] + split_commands(preparation.scan.build_commands),
            max_repairs=max(0, max_iterations - 1),
            last_synth_artifact=synth_artifact.content if synth_artifact else "",
        )

        final_execution = repair_result.final_execution
        tests_passed = bool(final_execution and final_execution.passed)
        patch_success = bool(repair_result.final_patch and repair_result.final_patch.success)
        build_validation = BuildValidator(repo_root=str(self.workspace.repo_root)).validate(
            [test_command_as_text(test_command)] + preparation.scan.build_commands
        )
        acceptance = AcceptanceValidator().validate(
            repo_root=str(self.workspace.repo_root),
            objective=objective,
            tests_passed=tests_passed and build_validation.passed,
            changed_files=self.git.changed_files(),
            expected_files=preparation.plan.files_to_create,
        )
        visual_validation = VisualValidator().validate(repo_root=str(self.workspace.repo_root))
        quality_score = QualityScorer().score(
            repo_root=str(self.workspace.repo_root),
            acceptance=acceptance,
            build=build_validation,
            visual=visual_validation,
            repair_attempts=repair_result.state.repair_count,
            changed_files=self.git.changed_files(),
        )
        diff = self.git.diff()
        feedback_text = self._repair_feedback(repair_result.to_dict())
        failure_type = (
            repair_result.state.last_failure_type.value
            if repair_result.state.last_failure_type
            else "success"
        )
        final_verdict = (
            convergence.final_verdict
            if tests_passed and build_validation.passed and acceptance.passed
            else "REJECTED - tests are failing or patch application failed."
        )
        result_payload = {
            "converged": repair_result.converged,
            "cognition_converged": convergence.converged,
            "iterations_run": convergence.iterations_run,
            "final_verdict": final_verdict,
            "patch_success": patch_success,
            "patch_path": self._first_patch_path(repair_result.to_dict()),
            "tests_passed": tests_passed,
            "return_code": final_execution.return_code if final_execution else -1,
            "feedback": feedback_text,
            "failure_type": failure_type,
            "diff": diff,
            "stdout": final_execution.stdout if final_execution else "",
            "stderr": final_execution.stderr if final_execution else "",
            "repository_scan": preparation.scan.to_dict(),
            "repository_context": preparation.context.to_dict(),
            "execution_plan": preparation.plan.to_dict(),
            "bootstrap": bootstrap_result.to_dict(),
            "repair_convergence": repair_result.to_dict(),
            "repair_attempts": repair_result.state.repair_count,
            "stop_reason": repair_result.state.stop_reason,
            "execution_branch": execution_branch,
            "changed_files": self.git.changed_files(),
            "acceptance": acceptance.to_dict(),
            "build_validation": build_validation.to_dict(),
            "visual_validation": visual_validation.to_dict(),
            "quality_score": quality_score.to_dict(),
            "task_plan": preparation.task_plan.to_dict() if preparation.task_plan else None,
            "execution_graph": preparation.execution_graph.to_dict() if preparation.execution_graph else None,
            "architecture_memory": preparation.architecture_memory.to_dict() if preparation.architecture_memory else None,
            "compressed_context": preparation.compressed_context.to_dict() if preparation.compressed_context else None,
            "objective_memory": self._record_objective_memory(
                objective=objective,
                preparation=preparation,
                repair_result=repair_result.to_dict(),
                outcome="passed" if tests_passed else "failed",
            ),
        }
        report = build_release_report(
            run_id=run_identifier,
            objective=objective,
            result=result_payload,
        )
        report_path = self.release_reports.write(report)
        result_payload["release_report"] = report.to_dict()
        result_payload["release_report_path"] = str(report_path)
        return result_payload

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
        generated_content = self.patch_parser.extract_file(
            coder_artifact.content,
            target_file,
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

    def _prepare_repository_execution(
        self,
        objective: str,
    ) -> RepositoryExecutionPreparation:
        return asyncio.run(self.repository_engine.prepare(objective))

    def _prepare_execution_branch(self, run_id: str | None) -> str | None:
        try:
            manager = GitManager(str(self.workspace.repo_root))
            if not manager.is_git_repository():
                return None
            return manager.create_execution_branch(run_id or "manual")
        except GitManagerError as exc:
            raise AutonomousRunError(f"Failed to prepare execution branch: {exc}") from exc

    def _generate_repair_patch(self, context: RepairContext) -> str:
        repair_result = self.loop.run_full(
            objective=context.to_prompt(),
            max_iterations=1,
        )
        if not repair_result.coder_artifacts:
            raise AutonomousRunError("Repair round did not produce a PRIMARY_CODER artifact.")
        return repair_result.coder_artifacts[-1].content

    def _record_objective_memory(
        self,
        *,
        objective: str,
        preparation: RepositoryExecutionPreparation,
        repair_result: dict,
        outcome: str,
    ) -> dict:
        final_patch = repair_result.get("final_patch") if isinstance(repair_result, dict) else None
        changed_files = []
        if isinstance(final_patch, dict):
            for item in final_patch.get("results", []):
                if isinstance(item, dict) and item.get("file_path"):
                    changed_files.append(str(item["file_path"]))
        failures = []
        state = repair_result.get("state") if isinstance(repair_result, dict) else None
        if isinstance(state, dict) and state.get("last_failure_type"):
            failures.append(str(state["last_failure_type"]))
        self.architecture_memory.record_outcome(
            repository_path=str(self.workspace.repo_root),
            modified_files=changed_files,
            failure_patterns=failures,
        )
        record = self.objective_memory.record(
            ObjectiveMemoryRecord(
                objective=objective,
                repository_path=str(self.workspace.repo_root),
                plan=preparation.task_plan.to_dict() if preparation.task_plan else preparation.plan.to_dict(),
                repairs=repair_result.get("repair_contexts", []) if isinstance(repair_result, dict) else [],
                failures=failures,
                outcome=outcome,
            )
        )
        return record.to_dict()

    @staticmethod
    def _ensure_target_allowed(
        preparation: RepositoryExecutionPreparation,
        target_file: str,
    ) -> None:
        if not target_file.strip():
            return
        allowed = set(preparation.plan.files_to_create) | set(preparation.plan.files_to_modify)
        if target_file not in allowed:
            preparation.plan.files_to_modify.append(target_file)

    @staticmethod
    def _first_patch_path(repair_result: dict) -> str | None:
        final_patch = repair_result.get("final_patch") or {}
        results = final_patch.get("results") if isinstance(final_patch, dict) else None
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                return first.get("resolved_path") or first.get("file_path")
        return None

    @staticmethod
    def _repair_feedback(repair_result: dict) -> str:
        state = repair_result.get("state", {})
        final_execution = repair_result.get("final_execution") or {}
        if state.get("success"):
            return "Execution succeeded after repair convergence."
        if final_execution:
            return (
                f"Execution failed. stop_reason={state.get('stop_reason')}\n\n"
                f"STDOUT:\n{final_execution.get('stdout', '')}\n\n"
                f"STDERR:\n{final_execution.get('stderr', '')}"
            )
        return f"Patch application failed. stop_reason={state.get('stop_reason')}"

    @staticmethod
    def _objective_with_repository_context(
        objective: str,
        preparation: RepositoryExecutionPreparation,
    ) -> str:
        return (
            f"{objective}\n\n"
            "REPOSITORY_EXECUTION_CONTEXT:\n"
            f"{preparation.to_prompt_context()}"
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


def test_command_as_text(command: list[str]) -> str:
    return " ".join(command)
