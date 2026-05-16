from __future__ import annotations
from datetime import datetime, timezone
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_exchange import ArtifactExchange
from backend.runtime.local_inference import LocalInference
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
    Responsibilities:
    - swap runtimes in correct sequence
    - call real inference per stage
    - persist each cognition artifact via exchange
    - shutdown active runtime after round completes
    - return ordered artifact list
    """

    CODER_ROLE = "PRIMARY_CODER"
    SYNTH_ROLE = "DEEPSEEK_SYNTH"
    JUDGE_ROLE = "JUDGE"

    CODER_PORT = 8000
    SYNTH_PORT = 8002
    JUDGE_PORT = 8001

    CODER_MODEL = "qwen-primary"
    SYNTH_MODEL = "deepseek-synth"
    JUDGE_MODEL = "qwen-judge"

    def __init__(
        self,
        *,
        swap_engine: RuntimeSwapEngine,
        exchange: ArtifactExchange,
        inference: LocalInference,
        coder_model: str = CODER_MODEL,
        synth_model: str = SYNTH_MODEL,
        judge_model: str = JUDGE_MODEL,
        coder_port: int = CODER_PORT,
        synth_port: int = SYNTH_PORT,
        judge_port: int = JUDGE_PORT,
    ) -> None:
        self.swap_engine = swap_engine
        self.exchange = exchange
        self.inference = inference
        self.coder_model = coder_model
        self.synth_model = synth_model
        self.judge_model = judge_model
        self.coder_port = coder_port
        self.synth_port = synth_port
        self.judge_port = judge_port

    def execute(
        self,
        *,
        objective: str,
        round_id: int = 1,
    ) -> list[CognitionArtifact]:
        """
        Execute a full courtroom cognition round with real inference.
        Stages:
        1. PRIMARY_CODER  — patch generation
        2. DEEPSEEK_SYNTH — architecture critique
        3. JUDGE          — convergence verdict
        Returns ordered list of 3 CognitionArtifacts.
        """
        if not objective.strip():
            raise AutonomousCourtroomError("objective must not be blank.")

        artifacts: list[CognitionArtifact] = []

        # ── Stage 1: PRIMARY_CODER ────────────────────────────────────────────
        coder_artifact = self._run_stage(
            role=self.CODER_ROLE,
            model=self.coder_model,
            port=self.coder_port,
            artifact_id=f"coder_round_{round_id}",
            round_id=round_id,
            objective=objective,
            prompt=(
                f"Generate an implementation strategy for: {objective}"
            ),
        )
        artifacts.append(coder_artifact)

        # ── Stage 2: DEEPSEEK_SYNTH ───────────────────────────────────────────
        synth_artifact = self._run_stage(
            role=self.SYNTH_ROLE,
            model=self.synth_model,
            port=self.synth_port,
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

        # ── Stage 3: JUDGE ────────────────────────────────────────────────────
        judge_artifact = self._run_stage(
            role=self.JUDGE_ROLE,
            model=self.judge_model,
            port=self.judge_port,
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

        # Shutdown active runtime after round completes
        self.swap_engine.shutdown_active()

        return artifacts

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
            raise AutonomousCourtroomError(f"rounds must be >= 1, got {rounds}.")

        all_rounds: list[list[CognitionArtifact]] = []
        for round_id in range(1, rounds + 1):
            round_artifacts = self.execute(
                objective=objective,
                round_id=round_id,
            )
            all_rounds.append(round_artifacts)
        return all_rounds

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_stage(
        self,
        *,
        role: str,
        model: str,
        port: int,
        artifact_id: str,
        round_id: int,
        objective: str,
        prompt: str,
        metadata: dict | None = None,
    ) -> CognitionArtifact:
        """Swap runtime, call inference, build artifact, persist, return."""
        process = RuntimeProcess(
            role=role,
            model=f"~/Forge/models/{model}",
            port=port,
        )
        self.swap_engine.swap(process)

        content = self.inference.infer(
            port=port,
            model=model,
            prompt=prompt,
        )

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