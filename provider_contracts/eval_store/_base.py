"""Abstract base class for eval-store providers (L7–L9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    """One item in an eval dataset."""

    id: str
    dataset: str
    input: dict[str, Any] = Field(default_factory=dict)
    expected_output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str = "approved"  # draft | approved | deprecated
    source_trace_id: str | None = None


class RunHandle(BaseModel):
    """Handle for an experiment / dataset run."""

    run_id: str
    dataset: str
    run_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    environment: str = "eval"


class ScoreRecord(BaseModel):
    """A numeric or categorical score attached to a run item or trace."""

    name: str
    value: float
    comment: str | None = None
    data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"] = "NUMERIC"
    trace_id: str | None = None
    item_id: str | None = None
    environment: str = "eval"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    """A telemetry observation / span fetched for ops scoring."""

    trace_id: str
    span_id: str = ""
    name: str = ""
    environment: str = "prod"
    start_time: datetime | None = None
    latency_ms: float = 0.0
    success: bool = True
    is_error: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


class ReviewLabel(BaseModel):
    """A human review label from an annotation queue."""

    queue: str
    trace_id: str
    item_id: str | None = None
    reviewer: str
    label: str
    rubric: str | None = None
    rubric_version: str | None = None
    comment: str | None = None
    created_at: datetime | None = None


class AbstractEvalStoreProvider(ABC):
    """Contract for offline eval datasets, runs, scores, traces, and review."""

    @abstractmethod
    async def upsert_dataset(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a named dataset."""

    @abstractmethod
    async def upsert_item(self, item: DatasetItem) -> None:
        """Create or update a dataset item (id is stable)."""

    @abstractmethod
    async def list_items(
        self,
        dataset: str,
        *,
        status: str | None = None,
    ) -> list[DatasetItem]:
        """List items in a dataset, optionally filtered by status."""

    @abstractmethod
    async def start_run(
        self,
        dataset: str,
        run_name: str,
        *,
        metadata: dict[str, Any] | None = None,
        environment: str = "eval",
    ) -> RunHandle:
        """Open a new experiment / dataset run."""

    @abstractmethod
    async def record_item_result(
        self,
        run: RunHandle,
        item_id: str,
        *,
        trace_id: str | None = None,
        scores: list[ScoreRecord] | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        """Record scores (and optional output) for one item in a run."""

    @abstractmethod
    async def upsert_score_config(
        self,
        name: str,
        version: str,
        *,
        data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"] = "CATEGORICAL",
        categories: list[str] | None = None,
        description: str | None = None,
    ) -> str:
        """Upsert a versioned score config (rubric). Returns config id/key."""

    @abstractmethod
    async def fetch_traces(
        self,
        *,
        environment: str,
        since: datetime | None = None,
        name: str | None = None,
        limit: int = 100,
    ) -> list[TraceRecord]:
        """Fetch observations/traces for operations scoring."""

    @abstractmethod
    async def enqueue_review(self, queue: str, trace_id: str) -> None:
        """Add a trace to an annotation / review queue."""

    @abstractmethod
    async def fetch_review_labels(self, queue: str) -> list[ReviewLabel]:
        """Fetch human labels from a review queue."""

    async def flush(self) -> None:
        """Flush buffered writes. Default no-op."""
        return None
