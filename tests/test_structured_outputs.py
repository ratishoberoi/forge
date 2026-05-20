import json

import pytest

from backend.runtime.structured_outputs import StructuredOutputError, validate_role_output


def test_primary_coder_contract_accepts_summary_and_files():
    payload = json.dumps({
        "summary": "Add calculator",
        "files": {"src/app.py": "print('ok')\n"},
    })

    normalized = validate_role_output("PRIMARY_CODER", payload)

    assert json.loads(normalized)["files"]["src/app.py"] == "print('ok')\n"


def test_primary_coder_contract_rejects_unsafe_path():
    payload = json.dumps({
        "summary": "bad path",
        "files": {"../app.py": "print('bad')"},
    })

    with pytest.raises(StructuredOutputError, match="Unsafe"):
        validate_role_output("PRIMARY_CODER", payload)


def test_synth_contract_rejects_missing_recommended_changes():
    payload = json.dumps({"critique": "x", "risks": []})

    with pytest.raises(StructuredOutputError, match="recommended_changes"):
        validate_role_output("DEEPSEEK_SYNTH", payload)


def test_judge_contract_requires_boolean_approved():
    payload = json.dumps({
        "verdict": "approved",
        "approved": "true",
        "required_changes": [],
    })

    with pytest.raises(StructuredOutputError, match="approved"):
        validate_role_output("JUDGE", payload)


def test_contract_rejects_reasoning_preamble():
    payload = 'I think this is fine.\n{"verdict":"ok","approved":true,"required_changes":[]}'

    with pytest.raises(StructuredOutputError, match="no prose"):
        validate_role_output("JUDGE", payload)
