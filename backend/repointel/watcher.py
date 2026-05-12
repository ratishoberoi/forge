"""Watchdog-based repository watcher with async callback bridging."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable

from backend.core.errors import ConfigurationError


class RepositoryWatcher:
    def __init__(
        self,
        root: str,
        on_change: Callable[[], Awaitable[None]],
        debounce_ms: int = 750,
    ) -> None:
        self._root = Path(root)
        self._on_change = on_change
        self._debounce_seconds = debounce_ms / 1000
        self._observer = None
        self._pending_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ModuleNotFoundError as exc:
            raise ConfigurationError(
                "watchdog is not installed. Install repository intelligence dependencies to enable file watching."
            ) from exc

        watcher = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event) -> None:  # type: ignore[override]
                if event.is_directory:
                    return
                watcher._schedule_callback()

        self._observer = Observer()
        self._observer.schedule(Handler(), str(self._root), recursive=True)
        self._observer.start()

    async def stop(self) -> None:
        if self._observer is None:
            return
        self._observer.stop()
        self._observer.join()
        self._observer = None

    def _schedule_callback(self) -> None:
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
        self._pending_task = asyncio.create_task(self._debounced_callback())

    async def _debounced_callback(self) -> None:
        await asyncio.sleep(self._debounce_seconds)
        await self._on_change()
