"""Shared provider error types."""


class ProviderError(Exception):
    """Raised when a provider call fails."""


class ProviderConfigError(ProviderError):
    """Raised when provider configuration is invalid or a collection size mismatches."""
