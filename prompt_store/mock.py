"""In-memory mock prompt store — backed by a plain dict, for dev/tests."""
from ._base import AbstractPromptStoreProvider


class MockPromptStoreProvider(AbstractPromptStoreProvider):
    """Dict-backed mock — pre-load prompts via constructor or ``set()``."""

    def __init__(self, prompts: dict[str, str] | None = None) -> None:
        self._store: dict[str, str] = dict(prompts) if prompts else {}

    def set(self, name: str, template: str) -> None:
        """Helper for test setup — not part of the ABC."""
        self._store[name] = template

    async def get(self, name: str, *, version: str | None = None) -> str:
        key = f"{name}:{version}" if version else name
        # Try versioned key first, then unversioned.
        if key in self._store:
            return self._store[key]
        if version and name in self._store:
            return self._store[name]
        raise KeyError(f"Prompt not found: {key}")

    async def list_names(self) -> list[str]:
        return list(self._store.keys())
