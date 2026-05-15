from __future__ import annotations
from backend.runtime.artifact import CognitionArtifact
from backend.runtime.artifact_context import ArtifactContextBuilder
from backend.runtime.artifact_loader import ArtifactLoader
from backend.runtime.artifact_store import ArtifactStore
from backend.runtime.live_adapter import LiveCognitionAdapter


class AutonomousArtifactOrchestrator:
    """
    Orchestrates multi-round cognition with artifact persistence.
    Responsibilities:
    - load prior artifacts for context
    - build enriched prompts from history
    - execute cognition via LiveCognitionAdapter
    - persist outputs as CognitionArtifacts
    - support multi-role round execution
    """

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        artifact_loader: ArtifactLoader,
        context_builder: ArtifactContextBuilder,
        adapter: LiveCognitionAdapter,
    ) -> None:
        self.artifact_store = artifact_store
        self.artifact_loader = artifact_loader
        self.context_builder = context_builder
        self.adapter = adapter

    async def execute_round(
        self,
        *,
        role: str,
        round_id: int,
        task: str,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        context_roles: list[str] | None = None,
    ) -> CognitionArtifact:
        """
        Execute a single cognition round for a given role.
        Loads prior artifacts, builds context, calls adapter, persists result.
        context_roles: which roles to include in prior context (default: same role only).
        """
        prior_artifacts = self._load_prior_context(role, context_roles)
        prior_context = self.context_builder.build_context(prior_artifacts)

        full_prompt = self._build_prompt(task, prior_context, prompt)

        response = await self.adapter.execute(
            role=role,
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        artifact = CognitionArtifact.create(
            role=role,
            round_id=round_id,
            task=task,
            content=response,
            metadata={
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        self.artifact_store.save(artifact)
        return artifact

    async def execute_multi_role_round(
        self,
        *,
        round_id: int,
        task: str,
        prompt: str,
        roles: list[str],
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> list[CognitionArtifact]:
        """
        Execute the same prompt across multiple roles in sequence.
        Each role sees only its own prior artifact history.
        Returns artifacts in role order.
        """
        artifacts: list[CognitionArtifact] = []
        for role in roles:
            artifact = await self.execute_round(
                role=role,
                round_id=round_id,
                task=task,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            artifacts.append(artifact)
        return artifacts

    async def execute_judge_round(
        self,
        *,
        round_id: int,
        task: str,
        coder_artifact: CognitionArtifact,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> CognitionArtifact:
        """
        Execute a JUDGE round that critiques a specific coder artifact.
        Injects coder content directly into the judge prompt.
        """
        judge_prompt = (
            f"TASK:\n{task}\n\n"
            f"CODER OUTPUT (round {coder_artifact.round_id}):\n"
            f"{coder_artifact.content}\n\n"
            f"Critique the above output. Be precise and constructive."
        )
        return await self.execute_round(
            role="JUDGE",
            round_id=round_id,
            task=task,
            prompt=judge_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _load_prior_context(
        self,
        role: str,
        context_roles: list[str] | None,
    ) -> list[CognitionArtifact]:
        """Load prior artifacts for given roles. Defaults to same role only."""
        roles = context_roles or [role]
        artifacts: list[CognitionArtifact] = []
        for r in roles:
            artifacts.extend(self.artifact_loader.load_role_artifacts(r))
        return sorted(artifacts, key=lambda a: (a.round_id, a.role))

    @staticmethod
    def _build_prompt(task: str, prior_context: str, prompt: str) -> str:
        """Assemble full LLM prompt from task, history, and current request."""
        parts = [f"TASK:\n{task}"]
        if prior_context.strip():
            parts.append(f"PRIOR COGNITION:\n{prior_context}")
        parts.append(f"CURRENT REQUEST:\n{prompt}")
        return "\n\n".join(parts)