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

    JSON_PATTERN = re.compile(r"\{", re.DOTALL)

    REQUIRED_FIELDS = ("summary", "reasoning", "risk", "files")
    PRIMARY_FIELDS = ("summary", "files")

    def extract_json(self, text: str) -> dict:
        """Extract and parse the first JSON object found in text."""
        decoder = json.JSONDecoder()
        for match in self.JSON_PATTERN.finditer(text):
            candidate = text[match.start():].strip()
            try:
                data, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data

        if "{" not in text:
            raise ValueError("No JSON payload found.")
        raise ValueError("Malformed JSON payload.")

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

    def parse_primary_output(self, text: str) -> ParsedPatchOutput:
        """Parse the strict PRIMARY_CODER contract used by AutonomousCourtroom."""
        data = self.extract_json(text)
        missing = [field for field in self.PRIMARY_FIELDS if field not in data]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        files = data["files"]
        if not isinstance(files, dict):
            raise ValueError("'files' must be a dict.")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in files.items()
        ):
            raise ValueError("'files' must be a dict[str, str].")

        return ParsedPatchOutput(
            summary=str(data["summary"]),
            reasoning=str(data.get("reasoning", "")),
            risk=str(data.get("risk", "unknown")),
            files=files,
        )

    def safe_parse_patch_output(
        self, text: str, default: ParsedPatchOutput | None = None
    ) -> ParsedPatchOutput | None:
        """Non-raising wrapper — returns default on any parse failure."""
        try:
            return self.parse_patch_output(text)
        except ValueError:
            try:
                return self.parse_primary_output(text)
            except ValueError:
                return default
