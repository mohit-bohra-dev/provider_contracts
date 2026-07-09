"""In-memory mock content safety provider — always safe, for dev/tests."""
from ._base import AbstractContentSafetyProvider, SafetyResult, SafetyVerdict


class MockContentSafetyProvider(AbstractContentSafetyProvider):
    """Always returns SAFE — useful for unit tests and local dev."""

    async def check(self, text: str) -> SafetyResult:
        return SafetyResult(
            verdict=SafetyVerdict.SAFE,
            score=0.0,
            reason=None,
            categories={},
        )
