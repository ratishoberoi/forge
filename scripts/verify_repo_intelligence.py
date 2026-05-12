"""End-to-end runtime verification for repository intelligence subsystems."""

from __future__ import annotations

import argparse
import asyncio
import sys

from backend.config.settings import Settings
from backend.repointel.diagnostics import RepositoryIntelligenceDiagnostics


async def main(repo_root: str | None) -> int:
    diagnostics = RepositoryIntelligenceDiagnostics(Settings())
    checks = await diagnostics.run(repo_root)
    failed = False
    for check in checks:
        line = f"[{check.status.upper():7}] {check.name:<16} {check.detail}"
        print(line)
        if check.status != "ok":
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.repo_root)))
