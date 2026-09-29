"""Postgres session store import smoke."""

from provider_contracts.session_store import (
    AbstractSessionStoreProvider,
    ConversationSession,
    ConversationTurn,
)
from provider_contracts.session_store.memory import InMemorySessionStoreProvider
from provider_contracts.session_store.postgres import PostgresSessionStoreProvider


def test_exports() -> None:
    assert AbstractSessionStoreProvider is not None
    assert ConversationSession is not None
    assert ConversationTurn is not None
    assert InMemorySessionStoreProvider is not None
    assert PostgresSessionStoreProvider is not None
