"""Abstract base class for cache providers."""
from abc import ABC, abstractmethod
from typing import Any


class AbstractCacheProvider(ABC):
    """Contract that every cache adapter must satisfy."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """
        Return the cached value for key, or None if missing / expired.

        Args:
            key: Cache key.
        """
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        """
        Store a value under key.

        Args:
            key: Cache key.
            value: Serialisable Python value.
            ttl: Time-to-live in seconds.  None means no expiry.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Evict the key from the cache."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True if the key exists and has not expired."""
        ...

    @abstractmethod
    async def clear(self, *, pattern: str = "*") -> int:
        """
        Remove all keys matching the glob pattern.

        Returns:
            Number of keys deleted.
        """
        ...
