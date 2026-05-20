from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json_store(path: Path, *, default: dict[str, Any], store_name: str) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        quarantine_corrupt_store(path, store_name=store_name, reason=str(exc))
        return dict(default)
    if not isinstance(data, dict):
        quarantine_corrupt_store(path, store_name=store_name, reason="root JSON value is not an object")
        return dict(default)
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def quarantine_corrupt_store(path: Path, *, store_name: str, reason: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine = path.with_name(f"{path.name}.corrupt-{timestamp}.bak")
    try:
        shutil.copy2(path, quarantine)
        path.unlink()
    except OSError:
        pass
    print(f"[MEMORY_CORRUPTION_RECOVERED] {store_name} path={path} backup={quarantine} reason={reason}")
    return quarantine
