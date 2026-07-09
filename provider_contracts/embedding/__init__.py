"""Embedding provider interfaces."""

from ._base import AbstractEmbeddingProvider, EmbeddingResult
from .gemini import GeminiEmbeddingProvider

__all__ = ["AbstractEmbeddingProvider", "EmbeddingResult", "GeminiEmbeddingProvider"]
