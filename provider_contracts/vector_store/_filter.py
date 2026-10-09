"""Shared metadata-filter helpers for vector store providers."""
from __future__ import annotations

import re
from typing import Any

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_filter_key(key: str) -> str:
    """Reject non-identifier metadata keys (injection / OData safety)."""
    if not _KEY_RE.match(key):
        raise ValueError(f"Invalid filter key: {key!r}")
    return key


def metadata_matches_filter(
    metadata: dict[str, Any], filter: dict[str, Any] | None
) -> bool:
    """Return True when ``metadata`` satisfies scalar equality / list any-of."""
    if not filter:
        return True
    for key, expected in filter.items():
        validate_filter_key(key)
        actual = metadata.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def odata_eq(key: str, value: Any) -> str:
    """Build a single OData equality clause (Azure AI Search)."""
    validate_filter_key(key)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"{key} eq '{escaped}'"
    if isinstance(value, bool):
        return f"{key} eq {str(value).lower()}"
    return f"{key} eq {value}"


def build_odata_filter(filter: dict[str, Any]) -> str:
    """Scalar → equality; list → any-of (OR of equalities)."""
    parts: list[str] = []
    for key, value in filter.items():
        validate_filter_key(key)
        if isinstance(value, list):
            if not value:
                parts.append("false")
            else:
                ors = [odata_eq(key, v) for v in value]
                parts.append("(" + " or ".join(ors) + ")")
        else:
            parts.append(odata_eq(key, value))
    return " and ".join(parts)
