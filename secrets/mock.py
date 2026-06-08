"""In-memory mock secrets provider — backed by a plain dict, for dev/tests."""
from ._base import AbstractSecretsProvider


class MockSecretsProvider(AbstractSecretsProvider):
    """Dict-backed mock — pre-load secrets via constructor or ``set()``."""

    def __init__(self, secrets: dict[str, str] | None = None) -> None:
        self._store: dict[str, str] = dict(secrets) if secrets else {}

    def set(self, name: str, value: str) -> None:
        """Helper for test setup — not part of the ABC."""
        self._store[name] = value

    async def get(self, name: str) -> str:
        try:
            return self._store[name]
        except KeyError:
            raise KeyError(f"Secret not found: {name}") from None

    async def get_or_default(self, name: str, default: str) -> str:
        return self._store.get(name, default)
