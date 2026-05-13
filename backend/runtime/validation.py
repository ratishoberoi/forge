from __future__ import annotations

from backend.runtime.patches import (
    StructuredPatch,
)


class PatchValidator:
    """
    Validates autonomous patch outputs.
    """

    def validate(
        self,
        patch: StructuredPatch,
    ) -> StructuredPatch:

        errors: list[str] = []

        if not patch.unified_diff.strip():
            errors.append(
                "Patch diff is empty."
            )

            return patch.model_copy(
                update={
                    "validation_errors": errors
                }
            )

        if (
            "diff --git"
            not in patch.unified_diff
        ):
            errors.append(
                "Patch is not a valid unified git diff."
            )

        if not patch.impacted_files:
            errors.append(
                "No impacted files detected."
            )

        return patch.model_copy(
            update={
                "validation_errors": errors
            }
        )