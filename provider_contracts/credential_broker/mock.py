"""In-memory credential broker for tests."""

from __future__ import annotations

from pydantic import SecretStr

from ._base import (
    AbstractCredentialBroker,
    AccessToken,
    CredentialCheck,
    CredentialDenied,
    CredentialNotConfigured,
    CredentialUnavailable,
    Principal,
)


class MockCredentialBroker(AbstractCredentialBroker):
    """In-memory broker for tests and contract tests."""

    def __init__(
        self,
        tokens: dict[str, str],
        denied: set[str] | frozenset[str] | None = None,
        unavailable: set[str] | frozenset[str] | None = None,
    ) -> None:
        self._tokens = dict(tokens)
        self._denied: frozenset[str] = frozenset(denied or ())
        self._unavailable: frozenset[str] = frozenset(unavailable or ())

    async def get_token(
        self,
        application_id: str,
        audience: str,
        *,
        principal: Principal | None = None,
    ) -> AccessToken:
        _ = principal
        if application_id in self._unavailable:
            raise CredentialUnavailable(
                application_id, f"Credential issuer unavailable for '{application_id}'"
            )
        if application_id in self._denied:
            raise CredentialDenied(
                application_id, f"Access denied for '{application_id}'"
            )
        value = self._tokens.get(application_id)
        if not value:
            raise CredentialNotConfigured(
                application_id,
                f"No credential configured for application '{application_id}'",
            )
        return AccessToken(
            application_id=application_id,
            audience=audience,
            token=SecretStr(value),
        )

    async def invalidate(self, application_id: str, audience: str) -> None:
        _ = (application_id, audience)

    async def check(self, application_id: str, audience: str) -> CredentialCheck:
        _ = audience
        if application_id in self._unavailable or application_id in self._denied:
            return CredentialCheck(
                application_id=application_id,
                state="error",
                detail="CredentialDenied"
                if application_id in self._denied
                else "CredentialUnavailable",
            )
        if application_id in self._tokens and self._tokens[application_id]:
            return CredentialCheck(application_id=application_id, state="configured")
        return CredentialCheck(application_id=application_id, state="missing")
