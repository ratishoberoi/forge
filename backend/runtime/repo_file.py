from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class RepositoryFile:
    path: str
    module: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path must not be empty.")
        if not self.module:
            raise ValueError("module must not be empty.")

    @property
    def extension(self) -> str:
        return self.path.rsplit(".", 1)[-1] if "." in self.path else ""

    @property
    def is_python(self) -> bool:
        return self.extension == "py"