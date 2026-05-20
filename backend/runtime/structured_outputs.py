from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any


class StructuredOutputError(ValueError):
    """Raised when a role response does not satisfy its JSON contract."""


def validate_role_output(role: str, text: str) -> str:
    candidates = _parse_json_candidates(text)
    if not candidates:
        raise StructuredOutputError("Response must contain a valid JSON object.")
    errors: list[StructuredOutputError] = []
    for data in candidates:
        try:
            normalized = _validate_role_data(role, data)
            return json.dumps(normalized, indent=2, sort_keys=True)
        except StructuredOutputError as exc:
            errors.append(exc)
    raise errors[0] if errors else StructuredOutputError("Response must contain a valid JSON object.")


def _validate_role_data(role: str, data: dict[str, Any]) -> dict[str, Any]:
    if role == "PRIMARY_CODER":
        return _validate_primary(data)
    elif role == "DEEPSEEK_SYNTH":
        return _validate_synth(data)
    elif role == "JUDGE":
        return _validate_judge(data)
    else:
        raise StructuredOutputError(f"Unknown courtroom role: {role}")


def _parse_json_candidates(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    candidates: list[dict[str, Any]] = []
    try:
        if stripped.startswith("{") and stripped.endswith("}"):
            data = json.loads(stripped)
            if isinstance(data, dict):
                candidates.append(data)
                return candidates
        else:
            candidates = _extract_json_objects(stripped)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError("Response is not valid JSON.") from exc
    return candidates


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            candidates.append(data)
    return candidates


def _validate_primary(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary")
    files = data.get("files")
    if not isinstance(summary, str) or not summary.strip():
        raise StructuredOutputError("PRIMARY_CODER.summary must be a non-empty string.")
    if not isinstance(files, dict) or not files:
        raise StructuredOutputError("PRIMARY_CODER.files must be a non-empty object.")
    normalized_files: dict[str, str] = {}
    for path, content in files.items():
        if not isinstance(path, str) or not isinstance(content, str):
            raise StructuredOutputError("PRIMARY_CODER.files must be {path: full_content}.")
        normalized_files[_validate_relative_path(path)] = content
    return {"summary": summary.strip(), "files": normalized_files}


def _validate_synth(data: dict[str, Any]) -> dict[str, Any]:
    critique = data.get("critique")
    risks = data.get("risks")
    recommended_changes = data.get("recommended_changes")
    if not isinstance(critique, str) or not critique.strip():
        raise StructuredOutputError("DEEPSEEK_SYNTH.critique must be a non-empty string.")
    return {
        "critique": critique.strip(),
        "risks": _validate_string_list(risks, "DEEPSEEK_SYNTH.risks"),
        "recommended_changes": _validate_string_list(
            recommended_changes,
            "DEEPSEEK_SYNTH.recommended_changes",
        ),
    }


def _validate_judge(data: dict[str, Any]) -> dict[str, Any]:
    verdict = data.get("verdict")
    approved = data.get("approved")
    required_changes = data.get("required_changes")
    if not isinstance(verdict, str) or not verdict.strip():
        raise StructuredOutputError("JUDGE.verdict must be a non-empty string.")
    if not isinstance(approved, bool):
        raise StructuredOutputError("JUDGE.approved must be a boolean.")
    return {
        "verdict": verdict.strip(),
        "approved": approved,
        "required_changes": _validate_string_list(
            required_changes,
            "JUDGE.required_changes",
        ),
    }


def _validate_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise StructuredOutputError(f"{field_name} must be a list.")
    if not all(isinstance(item, str) for item in value):
        raise StructuredOutputError(f"{field_name} must be a list[str].")
    return [item.strip() for item in value if item.strip()]


def _validate_relative_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise StructuredOutputError(f"Unsafe generated file path: {path}")
    return pure.as_posix()
