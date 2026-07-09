"""Environment-variable secrets provider."""
import os

from ._base import AbstractSecretsProvider


class EnvSecretsProvider(AbstractSecretsProvider):
    """Reads secrets from ``os.environ``.

    Suitable for local dev (via ``.env`` loaded by pydantic-settings)
    and simple container deployments where secrets are injected as env vars.
    """

    async def get(self, name: str) -> str:
        value = os.environ.get(name)
        if value is None:
            raise KeyError(f"Environment variable not set: {name}")
        return value

    async def get_or_default(self, name: str, default: str) -> str:
        return os.environ.get(name, default)
