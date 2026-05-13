from backend.runtime.patches import (
    PatchRisk,
    PatchTarget,
    StructuredPatch,
)
from backend.runtime.validation import (
    PatchValidator,
)


def build_patch(
    unified_diff: str,
    impacted_files: list[PatchTarget],
) -> StructuredPatch:
    return StructuredPatch(
        title="Test Patch",
        unified_diff=unified_diff,
        impacted_files=impacted_files,
        risk=PatchRisk.MEDIUM,
        summary="Test summary",
        reasoning="Test reasoning",
    )


def test_valid_patch():
    patch = build_patch(
        unified_diff="diff --git a/test.py b/test.py",
        impacted_files=[PatchTarget(path="test.py")],
    )
    validator = PatchValidator()
    validated = validator.validate(patch)
    assert validated.is_valid()
    assert validated.validation_errors == []


def test_empty_diff_invalid():
    patch = build_patch(
        unified_diff="",
        impacted_files=[PatchTarget(path="test.py")],
    )
    validator = PatchValidator()
    validated = validator.validate(patch)
    assert not validated.is_valid()
    assert "Patch diff is empty." in validated.validation_errors


def test_invalid_diff_format():
    patch = build_patch(
        unified_diff="hello world",
        impacted_files=[PatchTarget(path="test.py")],
    )
    validator = PatchValidator()
    validated = validator.validate(patch)
    assert not validated.is_valid()
    assert "Patch is not a valid unified git diff." in validated.validation_errors


def test_missing_impacted_files():
    patch = build_patch(
        unified_diff="diff --git a/test.py b/test.py",
        impacted_files=[],
    )
    validator = PatchValidator()
    validated = validator.validate(patch)
    assert not validated.is_valid()
    assert "No impacted files detected." in validated.validation_errors


def test_validation_does_not_mutate():
    patch = build_patch(unified_diff="", impacted_files=[])
    validator = PatchValidator()
    validated = validator.validate(patch)
    assert patch.validation_errors == []
    assert validated.validation_errors != []


def test_empty_diff_skips_file_check():
    patch = build_patch(unified_diff="", impacted_files=[])
    validator = PatchValidator()
    validated = validator.validate(patch)
    assert "Patch diff is empty." in validated.validation_errors
    assert "No impacted files detected." not in validated.validation_errors