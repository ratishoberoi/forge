"""
Forge autonomous execution entry point.

Prerequisites:
  Start ONE inference server before running:

  Terminal 1 (PRIMARY_CODER):
    python -m vllm.entrypoints.openai.api_server \
      --model ~/Forge/models/qwen-primary \
      --served-model-name qwen-primary \
      --port 8000

  Then run:
    python run_forge.py

  For full courtroom (all three runtimes):
    Terminal 2 (JUDGE):
      python -m vllm.entrypoints.openai.api_server \
        --model ~/Forge/models/qwen-judge \
        --served-model-name qwen-judge \
        --port 8001

    Terminal 3 (DEEPSEEK_SYNTH):
      python -m vllm.entrypoints.openai.api_server \
        --model ~/Forge/models/deepseek-synth \
        --served-model-name deepseek-synth \
        --port 8002
"""
from __future__ import annotations
import sys
from backend.runtime.autonomous_run import AutonomousRun, AutonomousRunError


def main() -> int:
    runner = AutonomousRun(
        repo_root=".",
        artifact_dir="./runtime_artifacts",
        inference_timeout=180,
        backup_writes=True,
    )

    try:
        runner.execute(
            objective=(
                "Improve authentication error handling in app.py. "
                "Add specific exception types, meaningful error messages, "
                "and ensure all auth failures are logged."
            ),
            target_file="sandbox_patch.py",
            test_command=["pytest", "-q"],
            max_iterations=2,
        )
        return 0

    except AutonomousRunError as exc:
        print(f"\n[FORGE ERROR] {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n[FORGE] Interrupted by user.")
        return 1
    except Exception as exc:
        print(f"\n[FORGE UNEXPECTED ERROR] {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())