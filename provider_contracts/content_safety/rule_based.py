"""Rule-based content safety provider — configurable keyword blocklist."""
from __future__ import annotations

from ._base import AbstractContentSafetyProvider, SafetyResult, SafetyVerdict

_DEFAULT_BLOCKLIST: frozenset[str] = frozenset({
    # Violence / threats
    "kill you",
    "bomb threat",
    "shoot up",
    "murder you",
    "attack you",
    "death threat",
    "weaponize",
    "blow up",
    # Self-harm
    "kill myself",
    "end my life",
    "commit suicide",
    "self-harm",
    "want to die",
    "cut myself",
    # Hate speech
    "hate all",
    "racial slur",
    "bigotry",
    "supremacist",
    # Sexual content
    "pornography",
    "explicit sexual",
    "xxx content",
    "erotica",
    # Jailbreak / prompt injection
    "ignore previous instructions",
    "ignore all instructions",
    "act as dan",
    "bypass restrictions",
    "jailbreak",
    "pretend you have no restrictions",
    "system override",
    "developer mode",
})


class RuleBasedContentSafetyProvider(AbstractContentSafetyProvider):
    """Keyword-based content safety — returns UNSAFE when any blocklist term is found.

    This is a lightweight stub for local dev. Use a real content-safety
    service (e.g. Azure AI Content Safety) in production.
    """

    def __init__(
        self,
        blocklist: frozenset[str] | set[str] | list[str] | None = None,
    ) -> None:
        self._blocklist: frozenset[str] = (
            frozenset(w.lower() for w in blocklist) if blocklist else _DEFAULT_BLOCKLIST
        )

    async def check(self, text: str) -> SafetyResult:
        text_lower = text.lower()
        matched: list[str] = [word for word in self._blocklist if word in text_lower]

        if matched:
            return SafetyResult(
                verdict=SafetyVerdict.UNSAFE,
                score=1.0,
                reason=f"Blocked terms found: {', '.join(sorted(matched))}",
                categories={term: 1.0 for term in matched},
            )

        return SafetyResult(
            verdict=SafetyVerdict.SAFE,
            score=0.0,
            reason=None,
            categories={},
        )
