"""Redis cache adapter using redis-py async client."""
import json
from typing import Any

import redis.asyncio as aioredis

from ._base import AbstractCacheProvider


class RedisCacheProvider(AbstractCacheProvider):
    """Cache backed by Redis using the official redis-py async client."""

    def __init__(self, url: str, *, key_prefix: str = "app:") -> None:
        self._client: aioredis.Redis[str] = aioredis.from_url(url, decode_responses=True)
        self._prefix = key_prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any | None:
        raw = await self._client.get(self._k(key))
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        serialised = json.dumps(value)
        if ttl is not None:
            await self._client.setex(self._k(key), ttl, serialised)
        else:
            await self._client.set(self._k(key), serialised)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._k(key))

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(self._k(key)))

    async def clear(self, *, pattern: str = "*") -> int:
        full_pattern = f"{self._prefix}{pattern}"
        keys: list[str] = await self._client.keys(full_pattern)
        if not keys:
            return 0
        deleted: int = await self._client.delete(*keys)
        return deleted
