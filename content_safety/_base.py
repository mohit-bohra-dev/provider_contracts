"""Abstract base class for content safety providers."""
from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel


class SafetyVerdict(StrEnum):
    """Outcome of a content safety check."""

    SAFE = "safe"
    UNSAFE = "unsafe"
    UNCERTAIN = "uncertain"


class SafetyResult(BaseModel):
    """Structured result from a content safety check."""

    verdict: SafetyVerdict
    score: float = 0.0
    """Severity score in range [0, 1]; 0 = definitely safe, 1 = definitely unsafe."""

    reason: str | None = None
    """Human-readable explanation (e.g. category that triggered)."""

    categories: dict[str, float] = {}
    """Per-category scores (provider-specific)."""


class AbstractContentSafetyProvider(ABC):
    """Contract that every content safety adapter must satisfy."""

    @abstractmethod
    async def check(self, text: str) -> SafetyResult:
        """
        Analyse *text* for unsafe content.

        Args:
            text: The content to evaluate.

        Returns:
            SafetyResult with verdict, score, and optional category breakdown.
        """
        ...
