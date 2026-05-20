from __future__ import annotations
from datetime import datetime, timezone
import time
import inspect
from collections.abc import Callable
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_exchange import ArtifactExchange
from backend.runtime.local_inference import LocalInference
from backend.runtime.runtime_health import RuntimeHealth
from backend.runtime.runtime_process import RuntimeProcess
from backend.runtime.runtime_swap_engine import RuntimeSwapEngine
from backend.runtime.structured_outputs import StructuredOutputError, validate_role_output
from backend.runtime.context_budget import ContextBudgetManager


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
    DEFAULT_CONTEXT_WINDOW = 8192
    DEFAULT_CONTEXT_SAFETY = 512

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
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        telemetry: Callable[[str], None] | None = None,
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
        self.context_window = context_window
        self.telemetry = telemetry or print
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
                launched = self.swap_engine.swap(process)
                self.port = launched.port
            except Exception as exc:
                last_error = str(exc)
                self._emit(
                    f"[SWAP] launch recovery {role} "
                    f"attempt {attempt}/{self.stage_attempts}: {exc}"
                )
                self._cleanup_active_runtime()
                continue

            ready = self._wait_until_runtime_ready(
                launched,
                role=role,
                model_name=model_name,
            )

            if ready:
                self._sync_context_window_from_runtime(launched)
                self._publish_runtime_status(launched, status="ready", failure_reason=None)
                self._emit(f"[READY] {role} healthy model={model_name}")
                self._emit(
                    f"[MODEL_READY] {role} model={model_name} port={launched.port}"
                )
                return launched

            last_error = (
                f"Runtime failed to become ready for role='{role}' "
                f"on port {launched.port} within {self.boot_timeout}s."
            )
            self._emit(
                f"[READY] {role} not ready attempt "
                f"{attempt}/{self.stage_attempts}"
            )
            self._emit(f"[MODEL_FAILED] {role} {last_error}")
            self._publish_runtime_status(
                launched,
                status="failed",
                failure_reason=last_error,
            )
            self._cleanup_active_runtime()

        raise AutonomousCourtroomError(last_error)

    def _wait_until_runtime_ready(
        self,
        launched: RuntimeProcess,
        *,
        role: str,
        model_name: str,
    ) -> bool:
        deadline = time.monotonic() + self.boot_timeout
        handle = launched.metadata.get("_popen")
        stderr_path = launched.metadata.get("stderr_path")
        if handle is None:
            return self._health_wait_until_ready(launched.port, model_name)
        while time.monotonic() < deadline:
            if self._health_is_ready(launched.port, model_name):
                return True
            if hasattr(handle, "poll"):
                return_code = handle.poll()
                if return_code is not None:
                    tail = ""
                    if isinstance(stderr_path, str):
                        tail = self.swap_engine.launcher._tail(stderr_path)
                    self._emit(
                        f"[MODEL_FAILED] {role} exited code={return_code} "
                        f"{tail[-500:]}"
                    )
                    return False
            time.sleep(min(self.health.poll_interval, 1.0))
        return False

    def _health_wait_until_ready(self, port: int, model_name: str) -> bool:
        try:
            return self.health.wait_until_ready(
                port=port,
                model_name=model_name,
                timeout=self.boot_timeout,
            )
        except TypeError:
            return self.health.wait_until_ready(
                port=port,
                timeout=self.boot_timeout,
            )

    def _health_is_ready(self, port: int, model_name: str) -> bool:
        try:
            return self.health.is_ready(port, model_name=model_name)
        except TypeError:
            return self.health.is_ready(port)

    def _publish_runtime_status(
        self,
        launched: RuntimeProcess,
        *,
        status: str,
        failure_reason: str | None,
    ) -> None:
        diagnostics = launched.metadata.get("runtime_diagnostics")
        if not isinstance(diagnostics, dict):
            return
        updated = {
            **diagnostics,
            "load_status": status,
            "failure_reason": failure_reason,
        }
        launched.metadata["runtime_diagnostics"] = updated
        launcher = self.swap_engine.launcher
        if hasattr(launcher, "last_diagnostics"):
            launcher.last_diagnostics = updated
        callback = getattr(launcher, "diagnostics_callback", None)
        if callback:
            callback(updated)

    def _sync_context_window_from_runtime(self, launched: RuntimeProcess) -> None:
        diagnostics = launched.metadata.get("runtime_diagnostics")
        if not isinstance(diagnostics, dict):
            return
        profile = diagnostics.get("profile")
        if not isinstance(profile, dict):
            return
        max_model_len = profile.get("max_model_len")
        if isinstance(max_model_len, int) and max_model_len > 0:
            self.context_window = max_model_len

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
        current_prompt = prompt
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
                    prompt=current_prompt,
                    metadata=metadata,
                )
            except StructuredOutputError as exc:
                last_error = exc
                self._emit(
                    f"[SCHEMA_VALIDATION] {role} failed attempt "
                    f"{attempt}/{self.inference_attempts}: {exc}"
                )
                current_prompt = self._schema_retry_prompt(
                    role=role,
                    original_prompt=prompt,
                    error=str(exc),
                )
                self._cleanup_active_runtime()
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
        system_prompt = self._system_prompt_for_role(role)
        max_tokens = self._max_tokens_for_role(role)
        prompt = self._budget_prompt(
            role=role,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        raw_content = self._infer_raw(
            role=role,
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        self._emit(f"[RAW_MODEL_OUTPUT] {role} chars={len(raw_content)} preview={raw_content[:180]!r}")
        recovered = _requires_json_recovery(raw_content)
        try:
            content = validate_role_output(role, raw_content)
        except StructuredOutputError as exc:
            self._emit(f"[SCHEMA_REPAIR] {role} {exc}")
            repair_max_tokens = self._repair_max_tokens_for_role(role, max_tokens)
            repair_prompt = self._schema_retry_prompt(
                role=role,
                original_prompt=prompt,
                error=str(exc),
                raw_output=raw_content,
            )
            repair_prompt = self._budget_prompt(
                role=role,
                prompt=repair_prompt,
                system_prompt=system_prompt,
                max_tokens=repair_max_tokens,
                safety_tokens=128,
            )
            self._emit(f"[SCHEMA_RETRY] {role} requesting strict JSON correction")
            repaired_content = self._infer_raw(
                role=role,
                model_name=model_name,
                prompt=repair_prompt,
                system_prompt=system_prompt,
                max_tokens=repair_max_tokens,
            )
            self._emit(
                f"[RAW_MODEL_OUTPUT] {role} repair_chars={len(repaired_content)} "
                f"preview={repaired_content[:180]!r}"
            )
            recovered = recovered or _requires_json_recovery(repaired_content)
            content = validate_role_output(role, repaired_content)
        if recovered:
            self._emit(f"[JSON_RECOVERY] {role} extracted top-level JSON object")
        self._emit(f"[SCHEMA_VALIDATION] {role} valid")
        self._emit(f"[INFER] {role} response received")

        artifact = CognitionArtifact(
            artifact_id=artifact_id,
            role=role,
            round_id=round_id,
            task=objective,
            content=content,
            created_at=datetime.now(timezone.utc),
            metadata={**(metadata or {}), "schema_valid": True},
        )

        self.exchange.persist(artifact)
        return artifact

    def _infer_raw(
        self,
        *,
        role: str,
        model_name: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
    ) -> str:
        kwargs = {
            "port": self.port,
            "model": model_name,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
        }
        if self._inference_accepts_response_format():
            kwargs["response_format"] = self._response_format_for_role(role)
        try:
            return self.inference.infer(**kwargs)
        except TypeError as exc:
            if "response_format" not in str(exc):
                raise
            kwargs.pop("response_format", None)
            return self.inference.infer(**kwargs)

    @staticmethod
    def _response_format_for_role(role: str) -> dict[str, str]:
        return {"type": "json_object"}

    def _inference_accepts_response_format(self) -> bool:
        try:
            signature = inspect.signature(self.inference.infer)
        except (TypeError, ValueError):
            return True
        return (
            "response_format" in signature.parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        )

    def _max_tokens_for_role(self, role: str) -> int:
        if role == self.CODER_ROLE:
            return min(900, max(LocalInference.DEFAULT_MAX_TOKENS, self.context_window // 3))
        return LocalInference.DEFAULT_MAX_TOKENS

    def _repair_max_tokens_for_role(self, role: str, current: int) -> int:
        if role == self.CODER_ROLE:
            return min(1600, max(current, self.context_window - 448))
        return current

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
            "Implement the requested functionality, not an objective summary, "
            "README-only change, health-check stub, or placeholder.\n"
            "Return ONLY a JSON object with no markdown and no prose:\n"
            "NO THINKING. NO ANALYSIS. NO EXPLANATION.\n"
            "Your response must start with { and end with }. "
            "Before returning, internally verify it parses as JSON. "
            "Do not include a preamble, epilogue, code fence, or explanation.\n"
            '{ "summary": "short summary", '
            '"files": { "relative/path.py": "complete file content" } }'
        )

    @staticmethod
    def _system_prompt_for_role(role: str) -> str:
        common = (
            "Return exactly one valid JSON object. Do not include markdown, "
            "backticks, comments, analysis, preamble, epilogue, or prose outside JSON. "
            "Do not output a thinking process. Do not explain your reasoning. "
            "The first character must be { and the final character must be }. "
            "Before returning, silently verify the response parses as JSON."
        )
        if role == AutonomousCourtroom.CODER_ROLE:
            return (
                common
                + ' Required schema: {"summary": string, '
                '"files": {"relative/path": "complete file content"}}. '
                "All file paths must be repository-relative. "
                "Do not satisfy application objectives with placeholders, "
                "objective_summary functions, README-only changes, or health-only stubs."
            )
        if role == AutonomousCourtroom.SYNTH_ROLE:
            return (
                common
                + ' Required schema: {"critique": string, "risks": string[], '
                '"recommended_changes": string[]}.'
            )
        return (
            common
            + ' Required schema: {"verdict": string, "approved": boolean, '
            '"required_changes": string[]}. Reject placeholders, README-only '
            "changes, objective_summary-only changes, health-only stubs, and "
            "outputs that do not implement the requested functionality."
        )

    def _schema_retry_prompt(
        self,
        *,
        role: str,
        original_prompt: str,
        error: str,
        raw_output: str = "",
    ) -> str:
        compact_task = self._compact_retry_task(role=role, original_prompt=original_prompt)
        return (
            "Your previous response violated the required output schema.\n"
            f"Schema error: {error}\n\n"
            "Return ONLY valid JSON. No prose. No markdown. No code fences. "
            "No thinking process. No analysis. No explanation. "
            "The response must start with { and end with }.\n\n"
            f"Task for {role}:\n{compact_task}"
        )

    def _compact_retry_task(self, *, role: str, original_prompt: str) -> str:
        if role != self.CODER_ROLE:
            return original_prompt[:2500]
        objective = self._extract_line_after(original_prompt, "Active objective:")
        paths = self._extract_path_mentions(original_prompt)
        path_text = "\n".join(f"- {path}" for path in paths[:16])
        return (
            f"Active objective: {objective or original_prompt[:300]}\n"
            "Return a PRIMARY_CODER payload with this exact schema:\n"
            "{\"summary\":\"non-empty summary\",\"files\":{\"path\":\"complete file content\"}}\n"
            "Create complete functional source and test files for every required application file.\n"
            "Required file paths:\n"
            f"{path_text}"
        )

    @staticmethod
    def _extract_line_after(text: str, prefix: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                return stripped[len(prefix):].strip()
        return ""

    @staticmethod
    def _extract_path_mentions(text: str) -> list[str]:
        candidates = [
            "requirements.txt",
            "pyproject.toml",
            "package.json",
            "index.html",
            "README.md",
            "Dockerfile",
            "calculator.py",
            "app/__init__.py",
            "app/main.py",
            "app/models.py",
            "app/database.py",
            "app/schemas.py",
            "app/repository.py",
            "tests/test_todos.py",
            "tests/test_app.py",
            "tests/test_calculator.py",
            "src/main.jsx",
            "src/App.jsx",
            "tests/app.test.mjs",
        ]
        return [path for path in candidates if path in text]

    def _budget_prompt(
        self,
        *,
        role: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        safety_tokens: int | None = None,
    ) -> str:
        budget = ContextBudgetManager()
        input_tokens = budget.estimate_tokens(prompt) + budget.estimate_tokens(system_prompt)
        total_tokens = input_tokens + max_tokens
        self._emit(
            f"[TOKEN_ESTIMATE] {role} input={input_tokens} "
            f"output={max_tokens} total={total_tokens} limit={self.context_window}"
        )
        allowed_input = self.context_window - max_tokens - (
            self.DEFAULT_CONTEXT_SAFETY if safety_tokens is None else safety_tokens
        )
        if input_tokens <= allowed_input:
            return prompt
        allowed_prompt_tokens = max(1, allowed_input - budget.estimate_tokens(system_prompt))
        allowed_chars = allowed_prompt_tokens * 4
        if len(prompt) <= allowed_chars:
            return prompt
        head_chars = int(allowed_chars * 0.72)
        tail_chars = max(0, allowed_chars - head_chars - 180)
        trimmed = (
            prompt[:head_chars]
            + "\n\n[... context trimmed to fit model context window ...]\n\n"
            + (prompt[-tail_chars:] if tail_chars else "")
        )
        self._emit(
            f"[CONTEXT_TRIMMED] {role} input={input_tokens} "
            f"allowed={allowed_input}"
        )
        return trimmed

    def _emit(self, message: str) -> None:
        self.telemetry(message)


def _requires_json_recovery(text: str) -> bool:
    stripped = text.strip()
    return not (stripped.startswith("{") and stripped.endswith("}"))
