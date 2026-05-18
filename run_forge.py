"""
Forge autonomous execution entry point.

Forge now manages runtime lifecycle automatically:
- launch
- health check
- inference
- shutdown
- model swap

No manual vLLM startup required.
"""

from __future__ import annotations

import sys
import traceback

from backend.runtime.autonomous_run import (
    AutonomousRun,
    AutonomousRunError,
)


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
                "Improve authentication "
                "error handling in app.py. "
                "Add specific exception types, "
                "meaningful error messages, "
                "and ensure all auth failures "
                "are logged."
            ),
            target_file="sandbox_patch.py",
            test_command=[
                "pytest",
                "-q",
            ],
            max_iterations=2,
        )

        return 0

    except AutonomousRunError as exc:
        print(
            f"\n[FORGE ERROR] {exc}"
        )

        traceback.print_exc()

        return 1

    except KeyboardInterrupt:
        print(
            "\n[FORGE] Interrupted by user."
        )

        return 1

    except Exception as exc:
        print(
            f"\n[FORGE UNEXPECTED ERROR] "
            f"{exc}"
        )

        traceback.print_exc()

        return 1

    finally:
        try:
            runner.loop.courtroom.swap_engine.shutdown_active()
        except Exception:
            pass

if __name__ == "__main__":
    sys.exit(main())