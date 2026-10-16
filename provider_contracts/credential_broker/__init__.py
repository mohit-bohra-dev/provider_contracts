"""Credential broker provider interfaces."""

from ._base import (
    AbstractCredentialBroker,
    AccessToken,
    CredentialCheck,
    CredentialDenied,
    CredentialError,
    CredentialNotConfigured,
    CredentialUnavailable,
    Principal,
    TokenSource,
    broker_token_source,
)
from .auth0_m2m import Auth0M2MCredentialBroker
from .mock import MockCredentialBroker
from .static import CachingCredentialBroker, StaticCredentialBroker

__all__ = [
    "AbstractCredentialBroker",
    "AccessToken",
    "Auth0M2MCredentialBroker",
    "CachingCredentialBroker",
    "CredentialCheck",
    "CredentialDenied",
    "CredentialError",
    "CredentialNotConfigured",
    "CredentialUnavailable",
    "MockCredentialBroker",
    "Principal",
    "StaticCredentialBroker",
    "TokenSource",
    "broker_token_source",
]
