from __future__ import annotations
import json
import re
from dataclasses import dataclass


@dataclass(slots=True)
class ParsedPatchOutput:
    summary: str
    reasoning: str
    risk: str
    files: dict[str, str]


class OutputParser:
    """
    Structured cognition output parser.
    Responsibilities:
    - extract JSON payloads
    - remove reasoning traces
    - validate structure
    - normalize outputs
    """

    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

    REQUIRED_FIELDS = ("summary", "reasoning", "risk", "files")

    def extract_json(self, text: str) -> dict:
        """Extract and parse the first JSON object found in text."""
        match = self.JSON_PATTERN.search(text)
        if not match:
            raise ValueError("No JSON payload found.")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed JSON payload.") from exc

    def parse_patch_output(self, text: str) -> ParsedPatchOutput:
        """Parse LLM output into a validated ParsedPatchOutput."""
        data = self.extract_json(text)

        missing = [f for f in self.REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        files = data["files"]
        if not isinstance(files, dict):
            raise ValueError("'files' must be a dict.")

        if not all(
            isinstance(k, str) and isinstance(v, str) for k, v in files.items()
        ):
            raise ValueError("'files' must be a dict[str, str].")

        return ParsedPatchOutput(
            summary=str(data["summary"]),
            reasoning=str(data["reasoning"]),
            risk=str(data["risk"]),
            files=files,
        )

    def safe_parse_patch_output(
        self, text: str, default: ParsedPatchOutput | None = None
    ) -> ParsedPatchOutput | None:
        """Non-raising wrapper — returns default on any parse failure."""
        try:
            return self.parse_patch_output(text)
        except ValueError:
            return default