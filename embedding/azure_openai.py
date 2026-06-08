"""Azure OpenAI embedding adapter."""
from __future__ import annotations

from openai import AsyncAzureOpenAI

from ._base import AbstractEmbeddingProvider, EmbeddingResult

_DEFAULT_MODEL = "text-embedding-3-large"
_DEFAULT_API_VERSION = "2024-08-01-preview"


class AzureOpenAIEmbeddingProvider(AbstractEmbeddingProvider):
    """Wraps the Azure OpenAI embeddings API.

    Requires the ``openai`` optional extra::

        pip install provider-contracts[openai]
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        deployment: str = _DEFAULT_MODEL,
        api_version: str = _DEFAULT_API_VERSION,
        *,
        azure_ad_token_provider=None,
    ) -> None:
        """Initialize the Azure OpenAI embedding provider.

        Args:
            endpoint: Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/)
            api_key: Azure OpenAI API key (optional if using managed identity)
            deployment: Deployment name for the embedding model
            api_version: Azure OpenAI API version
            azure_ad_token_provider: Token provider for managed identity authentication
        """
        # Configure client based on authentication method
        client_kwargs = {
            "azure_endpoint": endpoint,
            "api_version": api_version,
        }
        
        if azure_ad_token_provider:
            client_kwargs["azure_ad_token_provider"] = azure_ad_token_provider
        elif api_key:
            client_kwargs["api_key"] = api_key
        else:
            raise ValueError("Either api_key or azure_ad_token_provider must be provided")
            
        self._client = AsyncAzureOpenAI(**client_kwargs)
        self._deployment = deployment

    async def get_dimensions(self) -> int:
        res = await self.embed("")
        return res.dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate an embedding for a single text."""
        response = await self._client.embeddings.create(
            model=self._deployment,
            input=text,
        )
        vector: list[float] = response.data[0].embedding
        return EmbeddingResult(
            vector=vector,
            model=self._deployment,
            dimensions=len(vector),
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for a batch of texts."""
        response = await self._client.embeddings.create(
            model=self._deployment,
            input=texts,
        )
        return [
            EmbeddingResult(
                vector=item.embedding,
                model=self._deployment,
                dimensions=len(item.embedding),
            )
            for item in response.data
        ]