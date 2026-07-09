"""In-memory mock storage provider — zero I/O, for dev/tests."""
import time

from ._base import AbstractStorageProvider, StoredFile

_BASE_URL = "http://mock-storage"


class MockStorageProvider(AbstractStorageProvider):
    """Stores files in a plain dict in process memory.  Loses data on restart."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[bytes, str]] = {}  # key → (data, content_type)

    async def upload(
        self,
        data: bytes,
        key: str,
        *,
        content_type: str = "application/octet-stream",
        public: bool = False,
    ) -> StoredFile:
        self._store[key] = (data, content_type)
        return StoredFile(
            key=key,
            url=f"{_BASE_URL}/{key}",
            size_bytes=len(data),
            content_type=content_type,
        )

    async def download(self, key: str) -> bytes:
        if key not in self._store:
            raise FileNotFoundError(f"Mock storage: object not found: {key!r}")
        return self._store[key][0]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def get_url(self, key: str, *, expires_in: int = 3600) -> str:
        return f"{_BASE_URL}/{key}?expires={int(time.time()) + expires_in}"
