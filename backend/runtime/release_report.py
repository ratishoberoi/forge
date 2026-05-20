from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ReleaseReport:
    run_id: str
    objective: str
    plan: dict[str, Any]
    files_changed: list[str]
    tests: dict[str, Any]
    repairs: dict[str, Any]
    failures: list[str]
    verdict: str
    quality_score: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "plan": self.plan,
            "files_changed": self.files_changed,
            "tests": self.tests,
            "repairs": self.repairs,
            "failures": self.failures,
            "verdict": self.verdict,
            "quality_score": self.quality_score,
            "created_at": self.created_at,
        }


class ReleaseReportStore:
    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root or ".forge/reports").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, report: ReleaseReport) -> Path:
        path = self.root / f"{report.run_id}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def read(self, run_id: str) -> dict[str, Any]:
        path = self.root / f"{run_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))


def build_release_report(
    *,
    run_id: str,
    objective: str,
    result: dict[str, Any],
) -> ReleaseReport:
    repair = result.get("repair_convergence") if isinstance(result.get("repair_convergence"), dict) else {}
    state = repair.get("state") if isinstance(repair, dict) and isinstance(repair.get("state"), dict) else {}
    failure_type = result.get("failure_type")
    failures = [str(failure_type)] if failure_type and failure_type != "success" else []
    return ReleaseReport(
        run_id=run_id,
        objective=objective,
        plan=result.get("task_plan") or result.get("execution_plan") or {},
        files_changed=[str(path) for path in result.get("changed_files", [])],
        tests={
            "passed": result.get("tests_passed"),
            "return_code": result.get("return_code"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "build_validation": result.get("build_validation"),
            "acceptance": result.get("acceptance"),
            "visual_validation": result.get("visual_validation"),
        },
        repairs={
            "repair_attempts": result.get("repair_attempts", 0),
            "state": state,
        },
        failures=failures,
        verdict=str(result.get("final_verdict", "")),
        quality_score=result.get("quality_score") or {},
    )
