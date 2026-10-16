"""Auth0 client-credentials broker with injectable destination map."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

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


class Auth0M2MCredentialBroker(AbstractCredentialBroker):
    """Mint Auth0 client-credentials tokens with optional ``requested_destination``.

    ``destinations`` maps ``application_id`` → destination string. Audience is the
    shared API identifier passed at construction (or per-call override).
    """

    def __init__(
        self,
        *,
        token_url: str,
        client_id: str,
        client_secret: str,
        audience: str,
        destinations: dict[str, str],
    ) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._audience = audience
        self._destinations = dict(destinations)
        self._primed: set[str] = set()

    async def prime(self, application_ids: Sequence[str]) -> None:
        for app_id in application_ids:
            self._primed.add(app_id)

    async def get_token(
        self,
        application_id: str,
        audience: str,
        *,
        principal: Principal | None = None,
    ) -> AccessToken:
        _ = principal
        dest = self._destinations.get(application_id)
        if not dest:
            raise CredentialNotConfigured(
                application_id,
                f"No requested_destination mapped for application '{application_id}'",
            )
        aud = audience or self._audience
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise CredentialUnavailable(application_id, "httpx missing") from exc
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "audience": aud,
            "requested_destination": dest,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    self._token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
        except httpx.HTTPError as exc:
            raise CredentialUnavailable(
                application_id, f"Auth0 token request failed: {exc}"
            ) from exc
        if resp.status_code in {401, 403}:
            raise CredentialDenied(
                application_id,
                f"Auth0 refused token for destination '{dest}' ({resp.status_code})",
            )
        if resp.status_code >= 400:
            raise CredentialUnavailable(
                application_id,
                f"Auth0 token HTTP {resp.status_code} for destination '{dest}'",
            )
        payload = resp.json()
        token = str(payload.get("access_token") or "")
        if not token:
            raise CredentialUnavailable(application_id, "Auth0 returned empty access_token")
        expires_in = payload.get("expires_in")
        expires_at: datetime | None = None
        if isinstance(expires_in, (int, float)):
            expires_at = datetime.now(UTC) + timedelta(seconds=float(expires_in))
        return AccessToken(
            application_id=application_id,
            audience=aud,
            token=SecretStr(token),
            expires_at=expires_at,
        )

    async def invalidate(self, application_id: str, audience: str) -> None:
        _ = (application_id, audience)

    async def check(self, application_id: str, audience: str) -> CredentialCheck:
        _ = audience
        if application_id in self._destinations and self._client_id and self._client_secret:
            return CredentialCheck(application_id=application_id, state="configured")
        if application_id not in self._primed:
            return CredentialCheck(
                application_id=application_id, state="error", detail="unknown"
            )
        return CredentialCheck(application_id=application_id, state="missing")
