"""Contract tests for credential broker mock + caching."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from provider_contracts.credential_broker import (
    CachingCredentialBroker,
    CredentialNotConfigured,
    MockCredentialBroker,
)
from provider_contracts.credential_broker._base import AccessToken


@pytest.mark.asyncio
async def test_mock_get_token() -> None:
    broker = MockCredentialBroker({"app1": "tok-1"})
    token = await broker.get_token("app1", "aud")
    assert token.token.get_secret_value() == "tok-1"
    check = await broker.check("app1", "aud")
    assert check.state == "configured"


@pytest.mark.asyncio
async def test_mock_missing() -> None:
    broker = MockCredentialBroker({})
    with pytest.raises(CredentialNotConfigured):
        await broker.get_token("missing", "aud")


@pytest.mark.asyncio
async def test_caching_reuses_token() -> None:
    calls = {"n": 0}

    class Counting(MockCredentialBroker):
        async def get_token(  # type: ignore[override]
            self,
            application_id: str,
            audience: str,
            *,
            principal: object | None = None,
        ) -> AccessToken:
            calls["n"] += 1
            return AccessToken(
                application_id=application_id,
                audience=audience,
                token=SecretStr("cached"),
            )

    inner = Counting({})
    broker = CachingCredentialBroker(inner)
    t1 = await broker.get_token("a", "aud")
    t2 = await broker.get_token("a", "aud")
    assert t1.token.get_secret_value() == t2.token.get_secret_value() == "cached"
    assert calls["n"] == 1
