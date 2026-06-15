"""Contract tests for SessionStore."""

import asyncio
from datetime import datetime, timezone
import pytest

from packages.common.providers.session_store import AbstractSessionStoreProvider, ConversationTurn
from packages.common.providers.testing import MockSessionStoreProvider
from provider_contracts.session_store.memory import InMemorySessionStoreProvider

@pytest.fixture(params=["memory"])
def provider(request) -> AbstractSessionStoreProvider:
    if request.param == "memory":
        return InMemorySessionStoreProvider(ttl_minutes=10)

@pytest.mark.asyncio
async def test_create_and_get_session(provider: AbstractSessionStoreProvider) -> None:
    session = await provider.create_session("s1", rep_id="r1", loan_id="l1")
    assert session.session_id == "s1"
    assert session.rep_id == "r1"
    assert session.loan_id == "l1"
    assert len(session.turns) == 0

    fetched = await provider.get_session("s1")
    assert fetched is not None
    assert fetched.session_id == "s1"

@pytest.mark.asyncio
async def test_append_turn(provider: AbstractSessionStoreProvider) -> None:
    await provider.create_session("s2")
    
    turn = ConversationTurn(
        role="user",
        content="hello",
        timestamp=datetime.now(timezone.utc),
        turn_id="t1"
    )
    await provider.append_turn("s2", turn)
    
    fetched = await provider.get_session("s2")
    assert fetched is not None
    assert len(fetched.turns) == 1
    assert fetched.turns[0].content == "hello"

@pytest.mark.asyncio
async def test_ttl_expiration() -> None:
    provider = InMemorySessionStoreProvider(ttl_minutes=-1)
    await provider.create_session("s3")
    
    fetched = await provider.get_session("s3")
    assert fetched is None
