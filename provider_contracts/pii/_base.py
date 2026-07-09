"""Abstract base class for PII detection and anonymisation providers."""
from abc import ABC, abstractmethod

from pydantic import BaseModel


class PiiSpan(BaseModel):
    """A single detected PII entity within text."""

    start: int
    end: int
    entity_type: str
    """e.g. 'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'US_SSN'"""
    score: float
    """Detection confidence in range [0, 1]."""


class PiiResult(BaseModel):
    """Result of a PII detection pass."""

    original: str
    anonymised: str
    """Text with PII replaced by placeholders (e.g. '<PERSON>')."""
    entities: list[PiiSpan] = []


class AbstractPiiProvider(ABC):
    """Contract that every PII adapter must satisfy."""

    @abstractmethod
    async def anonymise(self, text: str) -> PiiResult:
        """
        Detect and redact PII from *text*.

        Args:
            text: Raw input that may contain PII.

        Returns:
            PiiResult with the anonymised text and detected entity spans.
        """
        ...
