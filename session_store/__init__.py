"""Session store provider contracts."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from packages.common.providers.session_store import AbstractSessionStoreProvider

__all__ = ["AbstractSessionStoreProvider"]
