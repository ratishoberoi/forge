"""Execution policy configuration for bounded subprocess validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExecutionPolicy:
    """Policy controlling subprocess execution bounds and output capture."""

    timeout_seconds: float = 30.0
    capture_output: bool = True
    max_output_chars: int = 20_000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0.")
        if self.max_output_chars < 0:
            raise ValueError("max_output_chars must be greater than or equal to 0.")
