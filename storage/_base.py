"""Abstract base class for file storage providers."""
from abc import ABC, abstractmethod

from pydantic import BaseModel


class StoredFile(BaseModel):
    """Metadata returned after a successful upload."""

    key: str
    """Provider-specific object key / path."""

    url: str
    """Public or pre-signed URL to access the file."""

    size_bytes: int
    content_type: str


class AbstractStorageProvider(ABC):
    """Contract that every storage adapter must satisfy."""

    @abstractmethod
    async def upload(
        self,
        data: bytes,
        key: str,
        *,
        content_type: str = "application/octet-stream",
        public: bool = False,
    ) -> StoredFile:
        """
        Upload raw bytes and return storage metadata.

        Args:
            data: File contents.
            key: Destination path / object key.
            content_type: MIME type of the file.
            public: Whether the file should be publicly accessible.
        """
        ...

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download and return raw bytes for the given key."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Permanently remove the object at key."""
        ...

    @abstractmethod
    async def get_url(self, key: str, *, expires_in: int = 3600) -> str:
        """
        Return an accessible URL for the given key.

        Args:
            key: Object key.
            expires_in: TTL in seconds for pre-signed URLs (ignored for public objects).
        """
        ...
