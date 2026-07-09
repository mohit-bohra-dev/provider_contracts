"""Abstract base class for prompt store providers."""
from abc import ABC, abstractmethod


class AbstractPromptStoreProvider(ABC):
    """
    Contract that every prompt store adapter must satisfy.

    A prompt store is a named registry of versioned prompt templates.
    Templates may contain placeholder variables rendered by the caller.
    """

    @abstractmethod
    async def get(self, name: str, *, version: str | None = None) -> str:
        """
        Retrieve a prompt template by name.

        Args:
            name: Logical prompt name (e.g. 'agent.system', 'tool.summary').
            version: Optional version tag; None returns the latest/default.

        Returns:
            The raw prompt template string.

        Raises:
            KeyError: If the prompt name (or version) does not exist.
        """
        ...

    @abstractmethod
    async def list_names(self) -> list[str]:
        """Return all known prompt names in this store."""
        ...
