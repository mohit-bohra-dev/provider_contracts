"""Google Gemini embedding adapter (google-genai SDK).

Supports single and batch embedding via the ``embed_content`` API.
The batch path passes a list of strings in a single call — the API returns
one embedding per item, preserving order.

Requires the ``gemini`` optional extra::

    pip install provider-contracts[gemini]
"""
from __future__ import annotations

from google import genai

from ._base import AbstractEmbeddingProvider, EmbeddingResult

_DEFAULT_MODEL = "text-embedding-004"


class GeminiEmbeddingProvider(AbstractEmbeddingProvider):
    """Wraps the Google Gemini Embedding API via ``google-genai``.

    Parameters
    ----------
    api_key:
        Google AI API key.
    model:
        Embedding model name (default ``text-embedding-004``).
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    # -- AbstractEmbeddingProvider interface ----------------------------------

    async def get_dimensions(self) -> int:
        """Probe the model for its embedding dimensionality."""
        result = await self.embed("")
        return result.dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate an embedding for a single text."""
        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
        )
        embeddings = response.embeddings
        if not embeddings:
            raise RuntimeError("Gemini embed_content returned no embeddings")
        values = embeddings[0].values
        if values is None:
            raise RuntimeError("Gemini embedding values are None")
        vector: list[float] = list(values)
        return EmbeddingResult(
            vector=vector,
            model=self._model,
            dimensions=len(vector),
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for a batch of texts in a single API call.

        The Gemini ``embed_content`` endpoint accepts a list of strings and
        returns one embedding per item, preserving input order.
        """
        if not texts:
            return []

        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=texts,
        )
        embeddings = response.embeddings
        if not embeddings:
            raise RuntimeError("Gemini embed_content returned no embeddings")
        results: list[EmbeddingResult] = []
        for emb in embeddings:
            values = emb.values
            if values is None:
                raise RuntimeError("Gemini embedding values are None")
            vec = list(values)
            results.append(
                EmbeddingResult(
                    vector=vec,
                    model=self._model,
                    dimensions=len(vec),
                )
            )
        return results
