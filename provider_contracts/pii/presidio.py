"""Presidio-based PII detection and anonymisation adapter."""
from __future__ import annotations

from typing import Any

from ._base import AbstractPiiProvider, PiiResult, PiiSpan

_DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "US_BANK_NUMBER",
    "DATE_TIME",
]


class PresidioPiiProvider(AbstractPiiProvider):
    """Wraps ``presidio-analyzer`` + ``presidio-anonymizer``.

    Requires the ``presidio`` optional extra::

        pip install provider-contracts[presidio]
    """

    def __init__(
        self,
        entities: list[str] | None = None,
        score_threshold: float = 0.5,
    ) -> None:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-untyped]
        from presidio_anonymizer import AnonymizerEngine  # type: ignore[import-untyped]

        self._analyzer: Any = AnalyzerEngine()
        self._anonymizer: Any = AnonymizerEngine()
        self._entities = entities or _DEFAULT_ENTITIES
        self._score_threshold = score_threshold

    async def anonymise(self, text: str) -> PiiResult:
        # Presidio is synchronous — run it inline (fast enough for per-turn use).
        results: Any = self._analyzer.analyze(
            text=text,
            entities=self._entities,
            language="en",
            score_threshold=self._score_threshold,
        )

        spans = [
            PiiSpan(
                start=r.start,
                end=r.end,
                entity_type=r.entity_type,
                score=r.score,
            )
            for r in results
        ]

        if not results:
            return PiiResult(original=text, anonymised=text, entities=[])

        anonymised_result: Any = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )

        return PiiResult(
            original=text,
            anonymised=anonymised_result.text,
            entities=spans,
        )
