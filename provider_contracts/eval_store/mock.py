"""In-memory mock eval store for unit tests."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from ._base import (
    AbstractEvalStoreProvider,
    DatasetItem,
    ReviewLabel,
    RunHandle,
    ScoreRecord,
    TraceRecord,
)


class MockEvalStoreProvider(AbstractEvalStoreProvider):
    """Fully in-memory EvalStore for assertions."""

    def __init__(self) -> None:
        self.datasets: dict[str, dict[str, Any]] = {}
        self.items: dict[str, list[DatasetItem]] = {}
        self.runs: list[RunHandle] = []
        self.results: list[dict[str, Any]] = []
        self.score_configs: dict[str, dict[str, Any]] = {}
        self.traces: list[TraceRecord] = []
        self.queues: dict[str, list[str]] = {}
        self.labels: dict[str, list[ReviewLabel]] = {}

    async def upsert_dataset(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.datasets[name] = {
            "description": description,
            "metadata": metadata or {},
        }
        self.items.setdefault(name, [])

    async def upsert_item(self, item: DatasetItem) -> None:
        bucket = self.items.setdefault(item.dataset, [])
        self.items[item.dataset] = [i for i in bucket if i.id != item.id] + [item]

    async def list_items(
        self,
        dataset: str,
        *,
        status: str | None = None,
    ) -> list[DatasetItem]:
        items = list(self.items.get(dataset, []))
        if status is not None:
            items = [i for i in items if i.status == status]
        return items

    async def start_run(
        self,
        dataset: str,
        run_name: str,
        *,
        metadata: dict[str, Any] | None = None,
        environment: str = "eval",
    ) -> RunHandle:
        handle = RunHandle(
            run_id=str(uuid.uuid4()),
            dataset=dataset,
            run_name=run_name,
            metadata=metadata or {},
            environment=environment,
        )
        self.runs.append(handle)
        return handle

    async def record_item_result(
        self,
        run: RunHandle,
        item_id: str,
        *,
        trace_id: str | None = None,
        scores: list[ScoreRecord] | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        self.results.append(
            {
                "run_id": run.run_id,
                "item_id": item_id,
                "trace_id": trace_id,
                "scores": scores or [],
                "output": output or {},
            }
        )

    async def upsert_score_config(
        self,
        name: str,
        version: str,
        *,
        data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"] = "CATEGORICAL",
        categories: list[str] | None = None,
        description: str | None = None,
    ) -> str:
        key = f"{name}@v{version}"
        self.score_configs[key] = {
            "name": name,
            "version": version,
            "data_type": data_type,
            "categories": categories or [],
            "description": description,
        }
        return key

    async def fetch_traces(
        self,
        *,
        environment: str,
        since: datetime | None = None,
        name: str | None = None,
        limit: int = 100,
    ) -> list[TraceRecord]:
        out: list[TraceRecord] = []
        for tr in self.traces:
            if tr.environment != environment:
                continue
            if name is not None and tr.name != name:
                continue
            if since is not None and tr.start_time is not None and tr.start_time < since:
                continue
            out.append(tr)
            if len(out) >= limit:
                break
        return out

    async def enqueue_review(self, queue: str, trace_id: str) -> None:
        self.queues.setdefault(queue, []).append(trace_id)

    async def fetch_review_labels(self, queue: str) -> list[ReviewLabel]:
        return list(self.labels.get(queue, []))
