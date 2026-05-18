from __future__ import annotations
import re


class PatchParser:
    """
    Extracts clean code content from model responses.
    Responsibilities:
    - strip markdown fenced code blocks
    - handle ```python and plain ``` fences
    - extract multiple blocks if needed
    - fall back to raw text if no fence found
    """

    FENCE_PATTERN = re.compile(
        r"```(?:python|py)?\n(.*?)```",
        re.DOTALL,
    )

    def extract_code(self, text: str) -> str:
        """
        Extract first fenced code block from text.
        Falls back to stripped raw text if no fence found.
        """
        if not text.strip():
            return ""

        fenced_blocks = self.FENCE_PATTERN.findall(text)

        if fenced_blocks:
            return fenced_blocks[0].strip()

        return text.strip()

    def extract_all_blocks(self, text: str) -> list[str]:
        """
        Extract all fenced code blocks from text.
        Returns empty list if none found.
        """
        return [block.strip() for block in self.FENCE_PATTERN.findall(text)]

    def has_code_block(self, text: str) -> bool:
        """Return True if text contains at least one fenced code block."""
        return bool(self.FENCE_PATTERN.search(text))

    def strip_preamble(self, text: str) -> str:
        """
        Remove any prose before the first code block.
        If no code block found, returns stripped text as-is.
        """
        match = self.FENCE_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()