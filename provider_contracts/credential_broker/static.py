"""Static secrets + caching credential brokers."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from pydantic import SecretStr

from provider_contracts.secrets import AbstractSecretsProvider

from ._base import (
    AbstractCredentialBroker,
    AccessToken,
    CredentialCheck,
    CredentialDenied,
    CredentialNotConfigured,
    CredentialUnavailable,
    Principal,
)


class StaticCredentialBroker(AbstractCredentialBroker):
    """Local/dev broker: ``APP_TOKEN_<APPID>`` secrets + optional shared token opt-in."""

    def __init__(
        self,
        secrets: AbstractSecretsProvider,
        shared_token: str = "",
        shared_apps: set[str] | frozenset[str] | None = None,
    ) -> None:
        self._secrets = secrets
        self._shared_token = shared_token
        self._shared_apps: frozenset[str] = frozenset(shared_apps or ())
        self._present: set[str] = set()
        self._errored: set[str] = set()
        self._primed: set[str] = set()

    async def prime(self, application_ids: Sequence[str]) -> None:
        for app_id in application_ids:
            self._primed.add(app_id)
            name = f"APP_TOKEN_{app_id.upper()}"
            try:
                value = await self._secrets.get_or_default(name, "")
            except Exception:  # noqa: BLE001 — presence probe must never raise
                self._errored.add(app_id)
                self._present.discard(app_id)
                continue
            if value:
                self._present.add(app_id)
                self._errored.discard(app_id)
            else:
                self._present.discard(app_id)

    async def get_token(
        self,
        application_id: str,
        audience: str,
        *,
        principal: Principal | None = None,
    ) -> AccessToken:
        _ = principal
        name = f"APP_TOKEN_{application_id.upper()}"
        value = await self._secrets.get_or_default(name, "")
        if value:
            return AccessToken(
                application_id=application_id,
                audience=audience,
                token=SecretStr(value),
            )
        if application_id in self._shared_apps and self._shared_token:
            return AccessToken(
                application_id=application_id,
                audience=audience,
                token=SecretStr(self._shared_token),
            )
        raise CredentialNotConfigured(
            application_id,
            f"No credential configured for application '{application_id}'",
        )

    async def invalidate(self, application_id: str, audience: str) -> None:
        _ = audience
        self._present.discard(application_id)

    async def check(self, application_id: str, audience: str) -> CredentialCheck:
        _ = audience
        if application_id in self._errored:
            return CredentialCheck(
                application_id=application_id, state="error", detail="secret_lookup_failed"
            )
        if application_id in self._present:
            return CredentialCheck(application_id=application_id, state="configured")
        if application_id in self._shared_apps and self._shared_token:
            return CredentialCheck(application_id=application_id, state="configured")
        if application_id not in self._primed:
            return CredentialCheck(
                application_id=application_id, state="error", detail="unknown"
            )
        return CredentialCheck(application_id=application_id, state="missing")


class CachingCredentialBroker(AbstractCredentialBroker):
    """Single-flight token cache wrapping an inner broker."""

    def __init__(self, inner: AbstractCredentialBroker, *, skew_seconds: float = 60.0) -> None:
        self._inner = inner
        self._skew = timedelta(seconds=skew_seconds)
        self._cache: dict[tuple[str, str, str], AccessToken] = {}
        self._locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._last_state: dict[tuple[str, str], CredentialCheck] = {}

    def _key(
        self, application_id: str, audience: str, principal: Principal | None
    ) -> tuple[str, str, str]:
        principal_key = principal.user_id if principal and principal.user_id else "-"
        return (application_id, audience, principal_key)

    def _fresh(self, token: AccessToken, now: datetime) -> bool:
        if token.expires_at is None:
            return True
        return token.expires_at - self._skew > now

    async def _lock_for(self, key: tuple[str, str, str]) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    async def get_token(
        self,
        application_id: str,
        audience: str,
        *,
        principal: Principal | None = None,
    ) -> AccessToken:
        key = self._key(application_id, audience, principal)
        now = datetime.now(UTC)
        cached = self._cache.get(key)
        if cached is not None and self._fresh(cached, now):
            return cached
        lock = await self._lock_for(key)
        async with lock:
            cached = self._cache.get(key)
            if cached is not None and self._fresh(cached, datetime.now(UTC)):
                return cached
            try:
                token = await self._inner.get_token(
                    application_id, audience, principal=principal
                )
            except CredentialNotConfigured as exc:
                self._last_state[(application_id, audience)] = CredentialCheck(
                    application_id=application_id, state="missing"
                )
                raise exc
            except CredentialDenied as exc:
                self._last_state[(application_id, audience)] = CredentialCheck(
                    application_id=application_id,
                    state="error",
                    detail="CredentialDenied",
                )
                raise exc
            except CredentialUnavailable as exc:
                self._last_state[(application_id, audience)] = CredentialCheck(
                    application_id=application_id,
                    state="error",
                    detail="CredentialUnavailable",
                )
                raise exc
            self._cache[key] = token
            self._last_state[(application_id, audience)] = CredentialCheck(
                application_id=application_id, state="configured"
            )
            return token

    async def invalidate(self, application_id: str, audience: str) -> None:
        drop = [k for k in self._cache if k[0] == application_id and k[1] == audience]
        for key in drop:
            del self._cache[key]
        await self._inner.invalidate(application_id, audience)

    async def check(self, application_id: str, audience: str) -> CredentialCheck:
        known = self._last_state.get((application_id, audience))
        if known is not None:
            return known
        for key, token in self._cache.items():
            if key[0] == application_id and key[1] == audience and self._fresh(
                token, datetime.now(UTC)
            ):
                return CredentialCheck(application_id=application_id, state="configured")
        return await self._inner.check(application_id, audience)

    async def prime(self, application_ids: Sequence[str]) -> None:
        await self._inner.prime(application_ids)
