"""Local filesystem storage adapter.

Files are stored under a configurable base directory.
URLs are served relative to a configurable base URL (e.g. a static file server or CDN).
"""
from pathlib import Path

from ._base import AbstractStorageProvider, StoredFile

_DEFAULT_BASE_DIR = Path("data/storage")
_DEFAULT_BASE_URL = "/static/storage"


class LocalStorageProvider(AbstractStorageProvider):
    """Persist files on the local filesystem — suitable for dev / single-node deployments."""

    def __init__(
        self,
        base_dir: str | Path = _DEFAULT_BASE_DIR,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._base_url = base_url.rstrip("/")

    def _full_path(self, key: str) -> Path:
        # Prevent path traversal
        full = (self._base_dir / key).resolve()
        if not str(full).startswith(str(self._base_dir.resolve())):
            raise ValueError(f"Key {key!r} escapes base directory")
        return full

    async def upload(
        self,
        data: bytes,
        key: str,
        *,
        content_type: str = "application/octet-stream",
        public: bool = False,
    ) -> StoredFile:
        path = self._full_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        url = f"{self._base_url}/{key}"
        return StoredFile(key=key, url=url, size_bytes=len(data), content_type=content_type)

    async def download(self, key: str) -> bytes:
        path = self._full_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Object not found: {key!r}")
        return path.read_bytes()

    async def delete(self, key: str) -> None:
        path = self._full_path(key)
        if path.exists():
            path.unlink()

    async def get_url(self, key: str, *, expires_in: int = 3600) -> str:
        # Local storage uses simple path-based URLs (no expiry)
        return f"{self._base_url}/{key}"
