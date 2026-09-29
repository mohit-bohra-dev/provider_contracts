"""Mock session store provider for tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

from provider_contracts.session_store._base import AbstractSessionStoreProvider


class MockSessionStoreProvider(AbstractSessionStoreProvider):
    """AsyncMock-backed session store for unit tests."""

    def __init__(self) -> None:
        self.create_session = AsyncMock()  # type: ignore[method-assign]
        self.get_session = AsyncMock()  # type: ignore[method-assign]
        self.append_turn = AsyncMock()  # type: ignore[method-assign]
        self.delete_session = AsyncMock()  # type: ignore[method-assign]
        self.close = AsyncMock()
