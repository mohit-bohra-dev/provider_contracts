"""In-memory mock PII provider — passthrough, for dev/tests."""
from ._base import AbstractPiiProvider, PiiResult


class MockPiiProvider(AbstractPiiProvider):
    """Passthrough — returns the original text unchanged, no entities detected."""

    async def anonymise(self, text: str) -> PiiResult:
        return PiiResult(
            original=text,
            anonymised=text,
            entities=[],
        )
