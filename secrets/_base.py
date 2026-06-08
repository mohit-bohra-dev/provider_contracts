"""Abstract base class for secrets management providers."""
from abc import ABC, abstractmethod


class AbstractSecretsProvider(ABC):
    """
    Contract that every secrets adapter must satisfy.

    Concrete implementations may read from .env files, environment variables,
    Azure Key Vault, AWS Secrets Manager, HashiCorp Vault, etc.
    """

    @abstractmethod
    async def get(self, name: str) -> str:
        """
        Retrieve a secret by name.

        Args:
            name: The secret identifier (e.g. 'OPENAI_API_KEY').

        Returns:
            The secret value as a string.

        Raises:
            KeyError: If the secret does not exist.
        """
        ...

    @abstractmethod
    async def get_or_default(self, name: str, default: str) -> str:
        """
        Retrieve a secret, returning *default* if not found.

        Args:
            name: The secret identifier.
            default: Fallback value.
        """
        ...
