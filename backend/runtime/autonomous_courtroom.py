from __future__ import annotations
from datetime import datetime, timezone
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_exchange import ArtifactExchange
from backend.runtime.local_inference import LocalInference
from backend.runtime.runtime_health import RuntimeHealth
from backend.runtime.runtime_process import RuntimeProcess
from backend.runtime.runtime_swap_engine import RuntimeSwapEngine


class AutonomousCourtroomError(Exception):
    """Raised when courtroom execution fails."""


class AutonomousCourtroom:
    """
    Integrated sequential cognition orchestration with real inference.
    Flow:
        PRIMARY_CODER  → generates patch strategy via real LLM
        DEEPSEEK_SYNTH → critiques architecture via real LLM
        JUDGE          → evaluates convergence via real LLM
    Key architecture:
    - ALL runtimes share port 8000
    - Previous runtime shuts down before next launches
    - RuntimeHealth waits until port ready before inferring
    - No assumption of pre-running servers
    """

    CODER_ROLE = "PRIMARY_CODER"
    SYNTH_ROLE = "DEEPSEEK_SYNTH"
    JUDGE_ROLE = "JUDGE"

    SHARED_PORT = 8000
    BOOT_TIMEOUT = 300
    DEFAULT_STAGE_ATTEMPTS = 2
    DEFAULT_INFERENCE_ATTEMPTS = 2

    CODER_MODEL_PATH = "~/Forge/models/qwen-primary"
    CODER_MODEL_NAME = "qwen-primary"

    SYNTH_MODEL_PATH = "~/Forge/models/deepseek-synth"
    SYNTH_MODEL_NAME = "deepseek-synth"

    JUDGE_MODEL_PATH = "~/Forge/models/qwen-judge"
    JUDGE_MODEL_NAME = "qwen-judge"

    def __init__(
        self,
        *,
        swap_engine: RuntimeSwapEngine,
        exchange: ArtifactExchange,
        inference: LocalInference,
        coder_model_path: str = CODER_MODEL_PATH,
        coder_model_name: str = CODER_MODEL_NAME,
        synth_model_path: str = SYNTH_MODEL_PATH,
        synth_model_name: str = SYNTH_MODEL_NAME,
        judge_model_path: str = JUDGE_MODEL_PATH,
        judge_model_name: str = JUDGE_MODEL_NAME,
        port: int = SHARED_PORT,
        boot_timeout: int = BOOT_TIMEOUT,
        stage_attempts: int = DEFAULT_STAGE_ATTEMPTS,
        inference_attempts: int = DEFAULT_INFERENCE_ATTEMPTS,
    ) -> None:
        self.swap_engine = swap_engine
        self.exchange = exchange
        self.inference = inference
        self.coder_model_path = coder_model_path
        self.coder_model_name = coder_model_name
        self.synth_model_path = synth_model_path
        self.synth_model_name = synth_model_name
        self.judge_model_path = judge_model_path
        self.judge_model_name = judge_model_name
        self.port = port
        self.boot_timeout = boot_timeout
        self.stage_attempts = max(1, stage_attempts)
        self.inference_attempts = max(1, inference_attempts)
        self.health = RuntimeHealth()

    def execute(
        self,
        *,
        objective: str,
        round_id: int = 1,
    ) -> list[CognitionArtifact]:
        """
        Execute a full courtroom cognition round with real inference.
        Stages:
        1. PRIMARY_CODER  — swap in, wait ready, generate patch
        2. DEEPSEEK_SYNTH — swap in, wait ready, critique patch
        3. JUDGE          — swap in, wait ready, evaluate verdict
        Returns ordered list of 3 CognitionArtifacts.
        """
        if not objective.strip():
            raise AutonomousCourtroomError("objective must not be blank.")

        artifacts: list[CognitionArtifact] = []

        try:
            # ── Stage 1: PRIMARY_CODER ────────────────────────────────────────
            coder_artifact = self._execute_stage(
                role=self.CODER_ROLE,
                model_path=self.coder_model_path,
                model_name=self.coder_model_name,
                artifact_id=f"coder_round_{round_id}",
                round_id=round_id,
                objective=objective,
                prompt=self._coder_prompt(objective),
            )
            artifacts.append(coder_artifact)

            # ── Stage 2: DEEPSEEK_SYNTH ───────────────────────────────────────
            synth_artifact = self._execute_stage(
                role=self.SYNTH_ROLE,
                model_path=self.synth_model_path,
                model_name=self.synth_model_name,
                artifact_id=f"synth_round_{round_id}",
                round_id=round_id,
                objective=objective,
                prompt=(
                    "Analyze architectural risks and repository impact "
                    f"for this patch:\n\n{coder_artifact.content}"
                ),
                metadata={"critiques_artifact": coder_artifact.artifact_id},
            )
            artifacts.append(synth_artifact)

            # ── Stage 3: JUDGE ────────────────────────────────────────────────
            judge_artifact = self._execute_stage(
                role=self.JUDGE_ROLE,
                model_path=self.judge_model_path,
                model_name=self.judge_model_name,
                artifact_id=f"judge_round_{round_id}",
                round_id=round_id,
                objective=objective,
                prompt=(
                    "Evaluate whether the following patch and architecture "
                    "critique appear safe and stable:\n\n"
                    f"PATCH:\n{coder_artifact.content}\n\n"
                    f"CRITIQUE:\n{synth_artifact.content}"
                ),
                metadata={
                    "reviews_coder": coder_artifact.artifact_id,
                    "reviews_synth": synth_artifact.artifact_id,
                },
            )
            artifacts.append(judge_artifact)

            return artifacts
        finally:
            self._cleanup_active_runtime()

    def execute_multi_round(
        self,
        *,
        objective: str,
        rounds: int = 3,
    ) -> list[list[CognitionArtifact]]:
        """
        Execute multiple sequential courtroom rounds.
        Each round's artifacts are persisted for next round's context.
        """
        if rounds < 1:
            raise AutonomousCourtroomError(
                f"rounds must be >= 1, got {rounds}."
            )
        all_rounds: list[list[CognitionArtifact]] = []
        for round_id in range(1, rounds + 1):
            round_artifacts = self.execute(
                objective=objective,
                round_id=round_id,
            )
            all_rounds.append(round_artifacts)
        return all_rounds

    # ── Internal ──────────────────────────────────────────────────────────────

    def _activate_runtime(
        self,
        *,
        role: str,
        model_path: str,
        model_name: str,
    ) -> RuntimeProcess:
        """
        Swap in next runtime and wait until ready.
        Previous runtime shut down by swap_engine before launch.
        Raises AutonomousCourtroomError if runtime does not boot in time.
        """
        last_error = ""
        for attempt in range(1, self.stage_attempts + 1):
            process = RuntimeProcess(
                role=role,
                model_path=model_path,
                model_name=model_name,
                port=self.port,
            )

            try:
                self.swap_engine.swap(process)
            except Exception as exc:
                last_error = str(exc)
                self._emit(
                    f"[SWAP] launch recovery {role} "
                    f"attempt {attempt}/{self.stage_attempts}: {exc}"
                )
                self._cleanup_active_runtime()
                continue

            ready = self.health.wait_until_ready(
                port=self.port,
                model_name=model_name,
                timeout=self.boot_timeout,
            )

            if ready:
                self._emit(f"[READY] {role} healthy model={model_name}")
                return process

            last_error = (
                f"Runtime failed to become ready for role='{role}' "
                f"on port {self.port} within {self.boot_timeout}s."
            )
            self._emit(
                f"[READY] {role} not ready attempt "
                f"{attempt}/{self.stage_attempts}"
            )
            self._cleanup_active_runtime()

        raise AutonomousCourtroomError(last_error)

    def _execute_stage(
        self,
        *,
        role: str,
        model_path: str,
        model_name: str,
        artifact_id: str,
        round_id: int,
        objective: str,
        prompt: str,
        metadata: dict | None = None,
    ) -> CognitionArtifact:
        last_error: Exception | None = None
        for attempt in range(1, self.inference_attempts + 1):
            self._activate_runtime(
                role=role,
                model_path=model_path,
                model_name=model_name,
            )
            try:
                return self._infer_and_persist(
                    role=role,
                    model_name=model_name,
                    artifact_id=artifact_id,
                    round_id=round_id,
                    objective=objective,
                    prompt=prompt,
                    metadata=metadata,
                )
            except Exception as exc:
                last_error = exc
                self._emit(
                    f"[INFER] {role} failed attempt "
                    f"{attempt}/{self.inference_attempts}: {exc}"
                )
                self._cleanup_active_runtime()

        raise AutonomousCourtroomError(
            f"Inference failed for role='{role}' after "
            f"{self.inference_attempts} attempts: {last_error}"
        )

    def _infer_and_persist(
        self,
        *,
        role: str,
        model_name: str,
        artifact_id: str,
        round_id: int,
        objective: str,
        prompt: str,
        metadata: dict | None = None,
    ) -> CognitionArtifact:
        """
        Call inference on currently active runtime, build artifact, persist.
        Uses explicit model_name — no more role-to-name mapping needed.
        """
        content = self.inference.infer(
            port=self.port,
            model=model_name,
            prompt=prompt,
            system_prompt=self._system_prompt_for_role(role),
        )
        self._emit(f"[INFER] {role} response received")

        artifact = CognitionArtifact(
            artifact_id=artifact_id,
            role=role,
            round_id=round_id,
            task=objective,
            content=content,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        self.exchange.persist(artifact)
        return artifact

    def _cleanup_active_runtime(self) -> None:
        try:
            self.swap_engine.shutdown_active()
        except Exception as exc:
            self._emit(f"[SWAP] cleanup failed: {exc}")

    @staticmethod
    def _coder_prompt(objective: str) -> str:
        return (
            "Generate the implementation for this objective:\n"
            f"{objective}\n\n"
            "Return ONLY valid JSON with this schema:\n"
            '{ "summary": str, "reasoning": str, "risk": str, '
            '"files": { "relative/path.py": "full file content" } }'
        )

    @staticmethod
    def _system_prompt_for_role(role: str) -> str:
        common = (
            "Return concise, valid JSON only. Do not use markdown fences. "
            "Do not include prose outside the JSON object."
        )
        if role == AutonomousCourtroom.CODER_ROLE:
            return (
                common
                + " The files object must map repository-relative paths to "
                "complete replacement file contents."
            )
        if role == AutonomousCourtroom.SYNTH_ROLE:
            return (
                common
                + ' Use keys "summary", "risks", "required_changes", '
                '"severity", and "verdict".'
            )
        return (
            common
            + ' Use keys "summary", "accepted", "blocking_issues", '
            '"confidence", and "verdict".'
        )

    @staticmethod
    def _emit(message: str) -> None:
        print(message)
