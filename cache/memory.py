"""In-process dictionary cache provider — for dev/tests (no Redis required)."""
import fnmatch
import time
from typing import Any

from ._base import AbstractCacheProvider


class MemoryCacheProvider(AbstractCacheProvider):
    """Simple dict-backed cache with optional TTL support."""

    def __init__(self) -> None:
        # key → (value, expires_at | None)
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _is_alive(self, key: str) -> bool:
        if key not in self._store:
            return False
        _, expires_at = self._store[key]
        if expires_at is not None and time.monotonic() > expires_at:
            del self._store[key]
            return False
        return True

    async def get(self, key: str) -> Any | None:
        if not self._is_alive(key):
            return None
        return self._store[key][0]

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        expires_at = time.monotonic() + ttl if ttl is not None else None
        self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return self._is_alive(key)

    async def clear(self, *, pattern: str = "*") -> int:
        keys_to_delete = [k for k in list(self._store) if fnmatch.fnmatch(k, pattern)]
        for k in keys_to_delete:
            del self._store[k]
        return len(keys_to_delete)
