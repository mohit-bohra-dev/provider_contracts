"""Mock session store provider for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

if TYPE_CHECKING:
    pass


class MockSessionStoreProvider:
    """Mock implementation of the SessionStoreProvider for testing."""

    def __init__(self) -> None:
        self.create_session = AsyncMock()
        self.get_session = AsyncMock()
        self.append_turn = AsyncMock()
        self.delete_session = AsyncMock()
        self.close = AsyncMock()

    # Need this so isinstance checks against Protocol might pass
    @property
    def __class__(self):
        # Return a type that satisfies the abstract protocol
        from packages.common.providers.session_store import AbstractSessionStoreProvider
        return AbstractSessionStoreProvider
