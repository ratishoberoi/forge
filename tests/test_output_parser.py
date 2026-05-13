import pytest
from backend.runtime.output_parser import OutputParser, ParsedPatchOutput


VALID_JSON = """
{
  "summary": "Add typing",
  "reasoning": "Improves clarity",
  "risk": "low",
  "files": {
    "hello.py": "def hello(name: str) -> str: return 'hello ' + name"
  }
}
"""


# ── extract_json ────────────────────────────────────────────────────────────

def test_extract_valid_json():
    parser = OutputParser()
    text = f"Thinking...\n{VALID_JSON}"
    data = parser.extract_json(text)
    assert data["summary"] == "Add typing"


def test_extract_json_no_json_raises():
    parser = OutputParser()
    with pytest.raises(ValueError, match="No JSON payload found"):
        parser.extract_json("garbage text with no braces")


def test_extract_json_malformed_raises():
    parser = OutputParser()
    with pytest.raises(ValueError, match="Malformed JSON"):
        parser.extract_json("{ bad json: !! }")


# ── parse_patch_output ──────────────────────────────────────────────────────

def test_parse_patch_output_returns_dataclass():
    parser = OutputParser()
    parsed = parser.parse_patch_output(VALID_JSON)
    assert isinstance(parsed, ParsedPatchOutput)


def test_parse_patch_output_fields():
    parser = OutputParser()
    parsed = parser.parse_patch_output(VALID_JSON)
    assert parsed.summary == "Add typing"
    assert parsed.reasoning == "Improves clarity"
    assert parsed.risk == "low"
    assert "hello.py" in parsed.files


def test_parse_patch_output_missing_fields_raises():
    parser = OutputParser()
    text = '{ "summary": "x" }'
    with pytest.raises(ValueError, match="Missing fields"):
        parser.parse_patch_output(text)


def test_parse_patch_output_files_not_dict_raises():
    parser = OutputParser()
    text = """
    {
      "summary": "x",
      "reasoning": "y",
      "risk": "low",
      "files": []
    }
    """
    with pytest.raises(ValueError, match="must be a dict"):
        parser.parse_patch_output(text)


def test_parse_patch_output_files_wrong_value_type_raises():
    parser = OutputParser()
    text = """
    {
      "summary": "x",
      "reasoning": "y",
      "risk": "low",
      "files": { "hello.py": 123 }
    }
    """
    with pytest.raises(ValueError, match="dict\\[str, str\\]"):
        parser.parse_patch_output(text)


def test_parse_patch_output_with_preamble():
    parser = OutputParser()
    text = f"Here is the patch output:\n{VALID_JSON}\nDone."
    parsed = parser.parse_patch_output(text)
    assert parsed.risk == "low"


# ── safe_parse_patch_output ─────────────────────────────────────────────────

def test_safe_parse_returns_none_on_failure():
    parser = OutputParser()
    result = parser.safe_parse_patch_output("no json here")
    assert result is None


def test_safe_parse_returns_default_on_failure():
    parser = OutputParser()
    fallback = ParsedPatchOutput(
        summary="fallback",
        reasoning="",
        risk="unknown",
        files={},
    )
    result = parser.safe_parse_patch_output("no json here", default=fallback)
    assert result is fallback


def test_safe_parse_returns_parsed_on_success():
    parser = OutputParser()
    result = parser.safe_parse_patch_output(VALID_JSON)
    assert isinstance(result, ParsedPatchOutput)
    assert result.summary == "Add typing"