"""JSONL file-based audit sink — one JSON line per event."""
from __future__ import annotations

import json
from pathlib import Path

from ._base import AbstractAuditSinkProvider, AuditEvent

_DEFAULT_DIR = "./audit"


class JsonlAuditSinkProvider(AbstractAuditSinkProvider):
    """Appends serialised ``AuditEvent`` records to a ``.jsonl`` file on disk.

    One file per provider instance; callers can rotate by creating a new
    instance with a different ``file_path`` or by using logrotate externally.
    """

    def __init__(
        self,
        directory: str = _DEFAULT_DIR,
        filename: str = "audit.jsonl",
    ) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / filename
        self._buffer: list[str] = []

    async def emit(self, event: AuditEvent) -> None:
        line = json.dumps(event.model_dump(), default=str)
        self._buffer.append(line)
        # Auto-flush every 50 events to balance I/O and data safety.
        if len(self._buffer) >= 50:
            await self.flush()

    async def flush(self) -> None:
        if not self._buffer:
            return
        with self._path.open("a", encoding="utf-8") as f:
            for line in self._buffer:
                f.write(line + "\n")
        self._buffer.clear()
