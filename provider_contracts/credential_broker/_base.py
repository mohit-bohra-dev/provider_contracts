"""Abstract base class for per-app credential brokers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr

TokenSource = Callable[[str, str], Awaitable[str]]


class CredentialError(Exception):
    """Base error for credential broker failures."""

    def __init__(self, application_id: str, message: str) -> None:
        self.application_id = application_id
        super().__init__(message)


class CredentialNotConfigured(CredentialError):
    """No access arrangement for this application."""


class CredentialDenied(CredentialError):
    """Issuer refused (revoked / not entitled)."""


class CredentialUnavailable(CredentialError):
    """Issuer unreachable."""


class AccessToken(BaseModel):
    """In-memory token; never persist or log ``token``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    application_id: str
    audience: str
    token: SecretStr
    token_type: Literal["Bearer"] = "Bearer"
    expires_at: datetime | None = None


class Principal(BaseModel):
    """Reserved for on-behalf-of flows. Unused by static/mock brokers."""

    user_id: str | None = None
    tenant: str | None = None
    subject_token: SecretStr | None = None


class CredentialCheck(BaseModel):
    application_id: str
    state: Literal["configured", "missing", "error"]
    detail: str | None = None


class AbstractCredentialBroker(ABC):
    """Per-app credential surface for invoke and spec refresh."""

    @abstractmethod
    async def get_token(
        self,
        application_id: str,
        audience: str,
        *,
        principal: Principal | None = None,
    ) -> AccessToken: ...

    @abstractmethod
    async def invalidate(self, application_id: str, audience: str) -> None: ...

    @abstractmethod
    async def check(self, application_id: str, audience: str) -> CredentialCheck:
        """Answer from in-memory state only — MUST NOT do network I/O."""

    async def prime(self, application_ids: Sequence[str]) -> None:
        """Optional warm-up (e.g. secret presence). Default no-op."""
        return None


def broker_token_source(broker: AbstractCredentialBroker) -> TokenSource:
    """Build a ``TokenSource`` that uses the broker; missing creds → empty string."""

    async def source(application_id: str, audience: str) -> str:
        try:
            access = await broker.get_token(application_id, audience)
        except CredentialNotConfigured:
            return ""
        except CredentialError:
            return ""
        return access.token.get_secret_value()

    return source
